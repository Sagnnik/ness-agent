"""Provider contracts shared by the runtime and command layer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

from langchain_core.language_models import BaseChatModel

from ness_cli.config import ProviderRuntimeConfig


@dataclass(frozen=True, slots=True)
class AuthState:
    authenticated: bool
    method: str = ""
    detail: str = ""


@dataclass(frozen=True, slots=True)
class AccountDetails:
    email: str | None = None
    tier: str | None = None


@dataclass(frozen=True, slots=True)
class RateLimitBucket:
    name: str
    window_minutes: int | None = None
    used_percent: float | None = None
    remaining_percent: float | None = None
    resets_at: int | None = None
    reached: bool = False


@dataclass(frozen=True, slots=True)
class ProviderStatus:
    provider: str
    auth: AuthState
    account: AccountDetails = field(default_factory=AccountDetails)
    limits: tuple[RateLimitBucket, ...] = ()
    credits: str | None = None
    warning: str | None = None


@dataclass(frozen=True, slots=True)
class ModelInfo:
    id: str
    name: str
    context_window: int | None = None
    input_price: float | None = None
    output_price: float | None = None
    cache_read_ratio: float = 0.1
    default_reasoning_effort: str | None = None
    reasoning_efforts: tuple[str, ...] = ()
    supports_vision: bool = False
    supports_anthropic_messages: bool = False
    is_default: bool = False


@dataclass(frozen=True, slots=True)
class LoginResult:
    status: Literal["complete", "pending", "cancelled", "error"]
    message: str
    auth_url: str | None = None
    login_id: str | None = None
    user_code: str | None = None
    verification_url: str | None = None


@dataclass(frozen=True, slots=True)
class LoginMethod:
    id: str
    label: str
    description: str = ""
    default: bool = False
    guidance: str | None = None
    input_kind: Literal["none", "secret"] = "none"
    input_label: str | None = None
    input_example: str = ""


class ProviderAdapter(ABC):
    """Boundary between provider protocols and the rest of the CLI."""

    id: str
    display_name: str
    login_description: str = ""
    selection_priority: int = 100
    billing_label: str = "unknown"

    def __init__(self, runtime_config: ProviderRuntimeConfig) -> None:
        self._runtime_config = runtime_config

    @property
    def runtime_config(self) -> ProviderRuntimeConfig:
        return self._runtime_config

    def reconfigure(self, runtime_config: ProviderRuntimeConfig) -> None:
        if runtime_config.provider_id != self.id:
            raise ValueError(
                f"cannot configure {self.id!r} with "
                f"{runtime_config.provider_id!r} settings"
            )
        self._runtime_config = runtime_config

    @abstractmethod
    def is_authenticated(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def build_chat_model(
        self,
        thread_id: str,
        *,
        model_name: str,
        reasoning_effort: str | None,
        session_suffix: str = "",
    ) -> BaseChatModel:
        raise NotImplementedError

    @abstractmethod
    async def models(self, *, refresh: bool = False) -> tuple[ModelInfo, ...]:
        raise NotImplementedError

    @abstractmethod
    def model_info(self, model_name: str) -> ModelInfo | None:
        raise NotImplementedError

    @abstractmethod
    async def status(self, *, refresh: bool = False) -> ProviderStatus:
        raise NotImplementedError

    def login_methods(self) -> tuple[LoginMethod, ...]:
        return ()

    async def login(
        self,
        *,
        method: str = "browser",
        secret: str | None = None,
    ) -> LoginResult:
        del method, secret
        return LoginResult(
            "error",
            f"{self.display_name} does not support interactive login.",
        )

    async def wait_for_login(self, login_id: str) -> LoginResult:
        del login_id
        return LoginResult(
            "error",
            f"{self.display_name} does not support interactive login.",
        )

    async def cancel_login(self, login_id: str) -> None:
        del login_id

    async def open_login_url(self, url: str) -> bool:
        del url
        return False

    async def logout(self) -> str:
        return f"{self.display_name} does not support logout."

    async def close(self) -> None:
        return None
