import json

import pytest

from ness_cli.mcp.config import ProjectMCPConfig, validate_import_entry
from ness_cli.mcp.importer import execute_mcp_import, plan_mcp_import


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ({"type": [], "command": "python"}, "type must be a string"),
        ({"command": "python", "url": None}, "both command and url"),
        (
            {"type": "stdio", "url": "https://example.com"},
            "stdio server requires command",
        ),
        ({"type": "http", "command": "python"}, "http server requires url"),
        ({"type": "sse", "url": "https://example.com"}, "unsupported transport"),
        ({"command": []}, "command must be a non-empty"),
        ({"command": ""}, "command must not be empty"),
        ({"command": ["python", 1]}, "command must be a non-empty"),
        ({"command": "python", "args": [1]}, "args must be an array"),
        ({"command": "python", "cwd": 1}, "cwd must be a string"),
        ({"command": "python", "envFile": []}, "envFile must be a string"),
        ({"command": "python", "description": []}, "description must be a string"),
        (
            {"command": "python", "startup_timeout": True},
            "startup_timeout must be a positive",
        ),
        (
            {"command": "python", "startup_timeout": 0},
            "startup_timeout must be a positive",
        ),
        ({"url": "https://example.com", "env": {"TOKEN": 1}}, "env must be an object"),
        ({"command": "python", "headers": []}, "headers must be an object"),
        (
            {"command": "python", "headersHelper": "helper"},
            "headersHelper is not supported",
        ),
        ({"command": "python", "oauth": {}}, "OAuth is supported only for HTTP"),
        (
            {"url": "https://example.com", "envFile": "secrets.env"},
            "envFile is only supported",
        ),
        (
            {"url": "https://example.com", "auth": {}, "oauth": {}},
            "both auth and oauth",
        ),
        ({"url": "https://example.com", "auth": []}, "auth must be an object"),
        ({"url": "https://example.com", "auth": {"CLIENT_ID": ""}}, "auth.CLIENT_ID"),
        (
            {
                "url": "https://example.com",
                "auth": {"CLIENT_ID": "id", "CLIENT_SECRET": []},
            },
            "auth.CLIENT_SECRET",
        ),
        (
            {"url": "https://example.com", "auth": {"CLIENT_ID": "id", "scopes": [1]}},
            "auth.scopes",
        ),
        ({"url": "https://example.com", "oauth": []}, "oauth must be an object"),
        ({"url": "https://example.com", "oauth": {"clientId": 1}}, "oauth.clientId"),
        (
            {"url": "https://example.com", "oauth": {"clientSecret": []}},
            "oauth.clientSecret",
        ),
        (
            {"url": "https://example.com", "oauth": {"callbackPort": True}},
            "oauth.callbackPort",
        ),
        (
            {"url": "https://example.com", "oauth": {"callbackPort": 65536}},
            "oauth.callbackPort",
        ),
        ({"url": "https://example.com", "oauth": {"scopes": None}}, "oauth.scopes"),
        (
            {
                "url": "https://example.com",
                "oauth": {"authServerMetadataUrl": "https://example.com"},
            },
            "authServerMetadataUrl is not supported",
        ),
        *[
            (
                {
                    "url": "https://example.com",
                    "oauth": {"tokenEndpointAuthMethod": method},
                },
                "oauth.tokenEndpointAuthMethod",
            )
            for method in ([], "invalid")
        ],
        (
            {
                "url": "https://example.com",
                "oauth": {},
                "headers": {"aUtHoRiZaTiOn": "token"},
            },
            "OAuth cannot be combined",
        ),
    ],
)
def test_import_and_load_agree_on_invalid_structure(tmp_path, entry, message):
    errors, _ = validate_import_entry(entry)
    assert any(message in error for error in errors)
    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps({"mcpServers": {"bad": entry, "good": {"command": "python"}}})
    )
    config = ProjectMCPConfig(path, project_root=tmp_path)
    assert set(config.specs) == {"good"}
    assert config.servers["bad"]["error"] == errors[0]


@pytest.mark.parametrize(
    "entry",
    [
        {"type": "", "command": "python", "env": None, "headers": None},
        {"command": ["python", "-m", "server"], "args": ["--flag"], "cwd": "."},
        {"type": "STREAMABLE-HTTP", "url": "https://example.com", "headers": None},
        {
            "url": "https://example.com",
            "auth": {"CLIENT_ID": "id", "scopes": "read write"},
        },
        {"url": "https://example.com", "oauth": {}},
        *[
            {"url": "https://example.com", "oauth": {"tokenEndpointAuthMethod": method}}
            for method in ("none", "client_secret_post", "client_secret_basic")
        ],
    ],
)
def test_import_and_load_accept_the_same_valid_structure(tmp_path, entry):
    assert validate_import_entry(entry)[0] == []
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"mcpServers": {"server": entry}}))
    config = ProjectMCPConfig(path, project_root=tmp_path)
    assert set(config.specs) == {"server"}


def test_import_retains_unavailable_placeholders_until_runtime(tmp_path, monkeypatch):
    monkeypatch.delenv("C01_MCP_ENDPOINT", raising=False)
    entry = {"url": "${C01_MCP_ENDPOINT}"}
    source, destination = tmp_path / "source.json", tmp_path / "mcp.json"
    source.write_text(json.dumps({"mcpServers": {"remote": entry}}))
    plan = plan_mcp_import(source, destination, project_root=tmp_path)
    assert plan.valid
    assert "unresolved placeholders" in " ".join(plan.entries[0].warnings)
    assert execute_mcp_import(plan, config_dir=tmp_path / "config") == []
    assert json.loads(destination.read_text())["mcpServers"]["remote"] == entry
    config = ProjectMCPConfig(destination, project_root=tmp_path)
    assert config.specs == {}
    assert "C01_MCP_ENDPOINT" in config.servers["remote"]["error"]
    monkeypatch.setenv("C01_MCP_ENDPOINT", "https://example.com/mcp")
    assert (
        ProjectMCPConfig(destination, project_root=tmp_path)
        .specs["remote"]
        .connection.url
        == "https://example.com/mcp"
    )


@pytest.mark.parametrize(
    ("url", "message"),
    [
        ("file:///tmp/server", "http(s)"),
        ("https://user:password@example.com/mcp", "embedded credentials"),
        ("https://example.com/mcp#fragment", "fragment"),
        ("https://example.com:invalid/mcp", "malformed"),
    ],
)
def test_runtime_validates_url_after_expansion(tmp_path, monkeypatch, url, message):
    entry = {"url": "${C01_ENDPOINT}"}
    assert validate_import_entry(entry)[0] == []
    literal_errors, _ = validate_import_entry({"url": url})
    assert any(message in error for error in literal_errors)
    monkeypatch.setenv("C01_ENDPOINT", url)
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"mcpServers": {"remote": entry}}))
    config = ProjectMCPConfig(path, project_root=tmp_path)
    assert config.specs == {}
    assert message in config.servers["remote"]["error"]
    assert "password" not in config.servers["remote"]["error"]


def test_import_collects_schema_errors_without_writing(tmp_path):
    entry = {
        "url": "https://example.com/mcp",
        "auth": {"CLIENT_ID": "", "CLIENT_SECRET": [], "scopes": [1]},
    }
    errors, _ = validate_import_entry(entry)
    assert len(errors) == 3
    assert all(
        any(field in error for error in errors)
        for field in ("auth.CLIENT_ID", "auth.CLIENT_SECRET", "auth.scopes")
    )
    source, destination = tmp_path / "source.json", tmp_path / "destination.json"
    source.write_text(json.dumps({"mcpServers": {"bad": entry}}))
    plan = plan_mcp_import(source, destination, project_root=tmp_path)
    assert not plan.valid
    assert not destination.exists()


@pytest.mark.parametrize("style", ["auth", "oauth"])
@pytest.mark.parametrize("secret", ["literal-secret", ""])
def test_oauth_defaults_and_scopes_follow_expansion(
    tmp_path, monkeypatch, style, secret
):
    monkeypatch.setenv("C01_CLIENT", "client-id")
    monkeypatch.setenv("C01_SECRET", secret)
    monkeypatch.setenv("C01_SCOPES", "read write")
    fields = (
        ("CLIENT_ID", "CLIENT_SECRET")
        if style == "auth"
        else ("clientId", "clientSecret")
    )
    entry = {
        "url": "https://example.com/mcp",
        style: {
            fields[0]: "${C01_CLIENT}",
            fields[1]: "${C01_SECRET}",
            "scopes": ["${C01_SCOPES}", "admin"],
        },
    }
    assert validate_import_entry(entry)[0] == []
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"mcpServers": {"remote": entry}}))
    config = ProjectMCPConfig(path, project_root=tmp_path)
    server = config.specs["remote"]
    assert server.oauth.client_id == "client-id"
    assert server.oauth.client_secret == secret
    assert server.oauth.scopes == ("read", "write", "admin")
    assert server.oauth.token_endpoint_auth_method == (
        "client_secret_post" if secret else "none"
    )
    assert server.oauth.callback_port == (8787 if style == "auth" else None)
    assert "literal-secret" not in " ".join(config.trust_preview.servers)
