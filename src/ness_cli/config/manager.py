from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ness_cli.config.model import (
    resolve_provider_runtime,
    resolve_runtime_config,
)
from ness_cli.config.settings import (
    Settings,
    load_documents,
    load_settings,
)
from ness_cli.config.store import ConfigStore, ConfigWriteError, SECRET_KEYS
from ness_cli.config.types import (
    CliOverrides,
    ConfigApplyError,
    ConfigPatch,
    ConfigUpdate,
    ProviderRuntimeConfig,
    RuntimeConfig,
)

if TYPE_CHECKING:
    from ness_cli.paths import NessPaths


_KEY_ALIASES: dict[str, str] = {
    "provider_id": "model_provider",
    "base_url": "openai_base_url",
    "session_id": "openrouter_session_id",
}

_PROFILE_KEYS = frozenset(
    {
        "model_name",
        "reflection_model_name",
        "reasoning_effort",
    }
)


def _apply_values(
    document: dict[str, Any],
    values: Mapping[str, Any],
) -> dict[str, Any]:
    for key, value in values.items():
        if value is None:
            document.pop(key, None)
        else:
            document[key] = value

    return document


def _apply_profile(
    document: dict[str, Any],
    provider_id: str,
    values: Mapping[str, Any],
) -> dict[str, Any]:
    profiles = document.get("provider_profiles")
    profiles = dict(profiles) if isinstance(profiles, dict) else {}

    profile = profiles.get(provider_id)
    profile = dict(profile) if isinstance(profile, dict) else {}

    for key, value in values.items():
        if value is None:
            profile.pop(key, None)
        else:
            profile[key] = value

    if profile:
        profiles[provider_id] = profile
    else:
        profiles.pop(provider_id, None)

    if profiles:
        document["provider_profiles"] = profiles
    else:
        document.pop("provider_profiles", None)

    return document


class ConfigManager:
    def __init__(
        self,
        *,
        store: ConfigStore,
        environment: Mapping[str, str],
        overrides: CliOverrides,
        sett: Settings,
        current: RuntimeConfig,
        provider: ProviderRuntimeConfig,
    ) -> None:
        self._store = store
        self._environment = dict(environment)
        self._overrides = overrides
        self._sett = sett
        self._current = current
        self._provider = provider

    @classmethod
    def load(
        cls,
        source: Path | NessPaths,
        overrides: CliOverrides | None = None,
        *,
        environment: Mapping[str, str] | None = None,
    ) -> ConfigManager:
        config_dir = getattr(source, "config_dir", source)
        store = ConfigStore(Path(config_dir))
        env = dict(os.environ if environment is None else environment)
        cli = overrides or CliOverrides()
        sett = load_settings(store, env)
        current, provider = resolve_runtime_config(
            sett,
            cli,
            env,
        )

        return cls(
            store=store,
            environment=env,
            overrides=cli,
            sett=sett,
            current=current,
            provider=provider,
        )

    @property
    def current(self) -> RuntimeConfig:
        return self._current

    @property
    def settings(self) -> Settings:
        """Return a detached settings snapshot for interactive editors."""
        return self._sett.model_copy(deep=True)

    def provider_profile(self, provider_id: str) -> dict[str, Any]:
        """Return a detached saved model profile for one provider."""
        value = self._sett.provider_profiles.get(provider_id)
        return dict(value) if isinstance(value, dict) else {}

    def provider_runtime(
        self,
        provider_id: str | None = None,
    ) -> ProviderRuntimeConfig:
        requested = provider_id or self._current.model.provider_id

        if requested == self._provider.provider_id:
            return self._provider

        return resolve_provider_runtime(
            self._sett,
            provider_id=requested,
            active_provider_id=self._current.model.provider_id,
            overrides=self._overrides,
        )

    def reload(self) -> RuntimeConfig:
        sett = load_settings(
            self._store,
            self._environment,
        )
        current, provider = resolve_runtime_config(
            sett,
            self._overrides,
            self._environment,
        )
        self._sett, self._current, self._provider = sett, current, provider
        return self._current

    def apply(self, patch: ConfigPatch) -> ConfigUpdate:
        requested = {
            _KEY_ALIASES.get(key, key): value for key, value in patch.values.items()
        }

        if not requested:
            return ConfigUpdate(
                previous=self._current,
                current=self._current,
                changed_keys=(),
                model_changed=False,
                provider_runtime_changed=False,
                options_changed=False,
                reload_selected=False,
                future_sessions_changed=False,
                restart_required=False,
                message="No configuration changes.",
            )

        allowed = set(Settings.model_fields)
        allowed.add("api_key")

        unknown = sorted(set(requested) - allowed)

        if unknown:
            names = ", ".join(unknown)
            raise ValueError(f"unknown configuration keys: {names}")

        previous = self._current
        previous_provider = self._provider

        requested_provider = requested.get("model_provider")

        if requested_provider is None and "model_provider" in requested:
            raise ValueError("model provider cannot be empty")

        target_provider = (
            str(requested_provider).strip()
            if requested_provider is not None
            else previous.model.provider_id
        )

        if not target_provider:
            raise ValueError("model provider cannot be empty")

        config_changes: dict[str, Any] = {}
        secret_changes: dict[str, Any] = {}
        profile_changes: dict[str, Any] = {}

        for key, value in requested.items():
            if key in _PROFILE_KEYS:
                profile_changes[key] = value
                continue

            if key == "api_key":
                secret_key = (
                    "opencode_api_key"
                    if target_provider == "opencode"
                    else "openai_api_key"
                )

                if target_provider == "codex":
                    raise ValueError(
                        "Codex authentication is managed by its login flow"
                    )

                secret_changes[secret_key] = value
                continue

            if key in SECRET_KEYS:
                secret_changes[key] = value
                continue

            config_changes[key] = value

        def prepare_documents(
            config: dict[str, Any], secrets: dict[str, Any]
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            config = _apply_values(config, config_changes)
            if profile_changes:
                config = _apply_profile(config, target_provider, profile_changes)
            secrets = _apply_values(secrets, secret_changes)
            sett = load_documents(config, secrets, self._environment)
            resolve_runtime_config(sett, self._overrides, self._environment)
            return config, secrets

        # Reject invalid requests before starting a save. Validate again against
        # the latest documents under the store's locks before either write.
        prepare_documents(self._store.load_config(), self._store.load_secrets())

        try:
            config, secrets = self._store.mutate_documents(prepare_documents)
            sett = load_documents(config, secrets, self._environment)
            current, provider = resolve_runtime_config(
                sett, self._overrides, self._environment
            )
        except Exception as write_error:
            failed_document = (
                write_error.document
                if isinstance(write_error, ConfigWriteError)
                else "configuration"
            )
            cause = (
                write_error.__cause__
                if isinstance(write_error, ConfigWriteError)
                else write_error
            )
            raise self._recover_failed_apply(
                patch, target_provider, previous, previous_provider, failed_document
            ) from cause

        self._sett, self._current, self._provider = sett, current, provider

        return self._build_update(
            previous, previous_provider, tuple(sorted(patch.values))
        )

    def _recover_failed_apply(
        self,
        patch: ConfigPatch,
        target_provider: str,
        previous: RuntimeConfig,
        previous_provider: ProviderRuntimeConfig,
        failed_document: str,
    ) -> ConfigApplyError:
        try:
            # A replacement can succeed before the writer raises. Failed reads
            # must not silently become defaults when reporting persisted values.
            config = self._store.load_config(strict=True)
            secrets = self._store.load_secrets(strict=True)
            sett = load_documents(config, secrets, self._environment)
            current, provider = resolve_runtime_config(
                sett, self._overrides, self._environment
            )
        except Exception as reload_error:
            error = ConfigApplyError(
                f"Error while saving {failed_document}. Could not reload persisted "
                "configuration; saved values are unknown.",
                update=None,
                saved_keys=None,
                unsaved_keys=None,
                failed_document=failed_document,
            )
            error.add_note(
                f"Configuration reload failed ({type(reload_error).__name__})."
            )
            return error

        self._sett, self._current, self._provider = sett, current, provider
        saved_keys = self._saved_keys(patch, target_provider, config, secrets)
        unsaved_keys = tuple(sorted(set(patch.values) - set(saved_keys)))
        message = (
            f"Error while saving {failed_document}. "
            f"Saved requested values: {', '.join(saved_keys) or 'none'}. "
            f"Not saved: {', '.join(unsaved_keys) or 'none'}."
        )
        update = self._build_update(
            previous, previous_provider, saved_keys, message=message
        )
        return ConfigApplyError(
            message,
            update=update,
            saved_keys=saved_keys,
            unsaved_keys=unsaved_keys,
            failed_document=failed_document,
        )

    @staticmethod
    def _saved_keys(
        patch: ConfigPatch,
        target_provider: str,
        config: Mapping[str, Any],
        secrets: Mapping[str, Any],
    ) -> tuple[str, ...]:
        profiles = config.get("provider_profiles")
        profile = profiles.get(target_provider) if isinstance(profiles, dict) else None
        profile = profile if isinstance(profile, dict) else {}
        saved = []
        for requested_key, value in patch.values.items():
            key = _KEY_ALIASES.get(requested_key, requested_key)
            if key in _PROFILE_KEYS:
                document = profile
            elif key == "api_key":
                key = (
                    "opencode_api_key"
                    if target_provider == "opencode"
                    else "openai_api_key"
                )
                document = secrets
            else:
                document = secrets if key in SECRET_KEYS else config
            if (value is None and key not in document) or (
                value is not None and key in document and document[key] == value
            ):
                saved.append(requested_key)
        return tuple(sorted(saved))

    def _build_update(
        self,
        previous: RuntimeConfig,
        previous_provider: ProviderRuntimeConfig,
        changed_keys: tuple[str, ...],
        *,
        message: str | None = None,
    ) -> ConfigUpdate:
        model_changed = previous.model != self._current.model
        provider_runtime_changed = previous_provider != self._provider
        options_changed = previous.settings != self._current.settings
        restart_required = previous.settings.ness_dir != self._current.settings.ness_dir
        reload_selected = model_changed or provider_runtime_changed
        future_sessions_changed = reload_selected or options_changed

        if not model_changed and not provider_runtime_changed and not options_changed:
            default_message = (
                "Saved configuration. An environment or CLI override "
                "still controls the current value."
            )
        elif restart_required:
            default_message = (
                "Saved configuration. Restart Ness to use the new "
                "project data directory."
            )
        else:
            default_message = "Configuration updated."

        return ConfigUpdate(
            previous=previous,
            current=self._current,
            changed_keys=changed_keys,
            model_changed=model_changed,
            provider_runtime_changed=provider_runtime_changed,
            options_changed=options_changed,
            reload_selected=reload_selected,
            future_sessions_changed=future_sessions_changed,
            restart_required=restart_required,
            message=default_message if message is None else message,
        )
