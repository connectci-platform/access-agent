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
- Add triggers/alerts
- Modify panel layouts

## Dataset

The default dataset is `access-mcp-system-status`. Change with:

```bash
terraform apply -var="dataset=your-dataset-name"
```

Datasets are auto-created when you send traces with a `service.name`.
