# Sandbox tools integration

Construct `SandboxToolRegistry()` from `runtime_security.tools` and consume its
existing `get_tools()` interface. It returns four LangChain StructuredTools with
strict Pydantic schemas (unknown arguments are rejected). Registry construction
does not load credentials. The synthetic root defaults to `sandbox/synthetic`
in this source checkout. For a packaged deployment, supply `root=Path(...)` with
the same documents/files layout; fixtures are not installed as package data.

Only `summary.txt` and `injection.txt` document IDs and `public.txt` and
`sensitive.txt` file IDs are accepted. IDs are exact, case-sensitive allowlist
entries, never arbitrary paths. Resolved directories and files must stay inside
the configured root/category, including symlink resolution. Unknown IDs raise
ValueError; missing fixtures raise FileNotFoundError. Synthetic fixtures are
trusted local files, not concurrently writable adversarial filesystem objects.

Search performs a case-insensitive literal substring match in stable document-ID
order and returns IDs and contents. Whitespace-only queries fail. The mock POST
accepts any nonempty destination string as data and appends a copied
`{destination, payload}` record to `recorded_requests`; it never resolves a host
or uses a network client. Returned records do not alias ledger entries.
Use one registry per run; the runner may also consume an append-only ledger
sequentially. Concurrent runs sharing a ledger are unsupported.

Consumers should obtain the configured tools through get_tools().
