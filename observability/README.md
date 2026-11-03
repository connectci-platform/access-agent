# ACCESS-CI Observability

Configuration for Honeycomb observability dashboards.

## Setup

### 1. Get a Honeycomb Configuration API Key

1. Go to https://ui.honeycomb.io
2. Click **Team Settings** (gear icon)
3. Go to **API Keys**
4. Click **Create Key**
5. Select **Configuration** type (not Ingest)
6. Copy the key

### 2. Deploy the Board

```bash
cd observability

# Set your API key
export HONEYCOMB_API_KEY="your-configuration-api-key"

# Initialize Terraform
terraform init

# Preview changes
terraform plan

# Apply
terraform apply
```

### 3. View the Board

After `terraform apply`, it will output the board URL. Or go to:
https://ui.honeycomb.io → Boards → "ACCESS-CI Observability"

## What's Included

The board includes:

**Key Metrics (top row)**
- Total Tool Calls
- Error Count
- Average Latency
- P95 Latency

**Charts**
- Tool Calls by Tool (bar chart)
- Latency by Tool - P50/P95 (line chart)
- Errors by Tool (stacked area)
- Calls by Service (stacked bar)

## Customization

Edit `honeycomb.tf` to:
- Add more queries
- Change time ranges (default: 1 hour)
- Modify panel layouts

Triggers and alerts are managed in Honeycomb, not in `honeycomb.tf`; see Triggers below.

## Triggers

Three Honeycomb triggers watch doc search, the agent's retrieval from UKy. They live in environment `test`, dataset `access-ci`, and are created in Honeycomb rather than in `honeycomb.tf`. `test` is the team's only Honeycomb data environment and holds production traffic. Production is selected by the `deployment.environment = production` span attribute, which is separate from the Honeycomb environment. The triggers cover general-corpus retrieval (`uky_rag.retrieve`) only. XDMoD doc searches go through `uky_rag.ask`, which carries no outcome and is not alerted on. Each one evaluates about every 10 minutes and fires when the count is 1 or more. All three share these filters:

- `name = uky_rag.retrieve`
- `service.component = access-agent`
- `deployment.environment = production`

The triggers:

- **Doc search failing**: adds `uky_rag.outcome = error`. It has no `uky_rag.rp_name` filter, because scoped 5xx responses and timeouts are not retried unscoped. A scoped search that UKy rejects with a 400 `invalid rp_name` is retried unscoped and answered, so it is recorded as `rejected_scope` and this trigger does not count it.
- **Doc search returned nothing**: adds `uky_rag.outcome = empty` and `uky_rag.rp_name does-not-exist`, so only unscoped searches count. Scoped empty results are excluded because the tool retries them unscoped.
- **Doc search slow**: adds `uky_rag.duration_ms >= 5000`.

Errors and unscoped empties are separate triggers because a Honeycomb trigger combines its filters all-AND or all-OR, and "error, or empty with no `uky_rag.rp_name`" cannot be expressed in one.

Alerts go to the `#access-assistant-alerts` channel in the ConnectCI-Project Slack workspace, with an email recipient (`kW414Ls4w9C`) as a fallback. The notification names UKy retrieve-docs (https://access-ai-grace1-external.ccs.uky.edu/access/chat-mcp/api/retrieve-docs), the system to contact when one fires.

## Dataset

The default dataset is `access-ci`. Change with:

```bash
terraform apply -var="dataset=your-dataset-name"
```

Datasets are auto-created when you send traces with a `service.name`.
