# Honeycomb Observability Configuration for ACCESS-CI
#
# Usage:
#   1. Set HONEYCOMB_API_KEY environment variable
#   2. terraform init
#   3. terraform apply

terraform {
  required_providers {
    honeycombio = {
      source  = "honeycombio/honeycombio"
      version = "~> 0.45"
    }
  }
}

provider "honeycombio" {
  # API key from HONEYCOMB_API_KEY env var
}

# Variables
variable "dataset" {
  description = "Honeycomb dataset name for all ACCESS-CI services"
  type        = string
  default     = "access-ci"
}

# ============================================================================
# Flexible Board: ACCESS-CI Observability
# ============================================================================

resource "honeycombio_flexible_board" "access_observability" {
  name        = "ACCESS-CI Observability"
  description = "MCP server and agent monitoring for ACCESS-CI"

  # Row 1: Key Metrics
  panel {
    type = "query"
    position {
      x_coordinate = 0
      y_coordinate = 0
      width        = 6
      height       = 4
    }
    query_panel {
      query_id            = honeycombio_query.tool_call_count.id
      query_annotation_id = honeycombio_query_annotation.tool_call_count.id
      query_style         = "graph"
    }
  }

  panel {
    type = "query"
    position {
      x_coordinate = 6
      y_coordinate = 0
      width        = 6
      height       = 4
    }
    query_panel {
      query_id            = honeycombio_query.error_count.id
      query_annotation_id = honeycombio_query_annotation.error_count.id
      query_style         = "graph"
    }
  }

  panel {
    type = "query"
    position {
      x_coordinate = 12
      y_coordinate = 0
      width        = 6
      height       = 4
    }
    query_panel {
      query_id            = honeycombio_query.avg_latency.id
      query_annotation_id = honeycombio_query_annotation.avg_latency.id
      query_style         = "graph"
    }
  }

  panel {
    type = "query"
    position {
      x_coordinate = 18
      y_coordinate = 0
      width        = 6
      height       = 4
    }
    query_panel {
      query_id            = honeycombio_query.p95_latency.id
      query_annotation_id = honeycombio_query_annotation.p95_latency.id
      query_style         = "graph"
    }
  }

  # Row 2: Tool Distribution
  panel {
    type = "query"
    position {
      x_coordinate = 0
      y_coordinate = 4
      width        = 12
      height       = 8
    }
    query_panel {
      query_id            = honeycombio_query.tool_call_by_tool.id
      query_annotation_id = honeycombio_query_annotation.tool_call_by_tool.id
      query_style         = "combo"
    }
  }

  panel {
    type = "query"
    position {
      x_coordinate = 12
      y_coordinate = 4
      width        = 12
      height       = 8
    }
    query_panel {
      query_id            = honeycombio_query.latency_by_tool.id
      query_annotation_id = honeycombio_query_annotation.latency_by_tool.id
      query_style         = "graph"
    }
  }

  # Row 3: Errors and Services
  panel {
    type = "query"
    position {
      x_coordinate = 0
      y_coordinate = 12
      width        = 12
      height       = 8
    }
    query_panel {
      query_id            = honeycombio_query.errors_by_tool.id
      query_annotation_id = honeycombio_query_annotation.errors_by_tool.id
      query_style         = "combo"
    }
  }

  panel {
    type = "query"
    position {
      x_coordinate = 12
      y_coordinate = 12
      width        = 12
      height       = 8
    }
    query_panel {
      query_id            = honeycombio_query.calls_by_service.id
      query_annotation_id = honeycombio_query_annotation.calls_by_service.id
      query_style         = "combo"
    }
  }
}

# ============================================================================
# Queries
# ============================================================================

resource "honeycombio_query" "tool_call_count" {
  dataset = var.dataset

  query_json = jsonencode({
    calculations = [{ op = "COUNT" }]
    filters = [
      { column = "mcp.method.name", op = "=", value = "tools/call" }
    ]
    time_range = 3600
  })
}

resource "honeycombio_query_annotation" "tool_call_count" {
  dataset     = var.dataset
  query_id    = honeycombio_query.tool_call_count.id
  name        = "Total Tool Calls"
  description = "Count of all MCP tool calls"
}

resource "honeycombio_query" "error_count" {
  dataset = var.dataset

  query_json = jsonencode({
    calculations = [{ op = "COUNT" }]
    filters = [
      { column = "status_code", op = "=", value = "STATUS_CODE_ERROR" }
    ]
    time_range = 3600
  })
}

resource "honeycombio_query_annotation" "error_count" {
  dataset     = var.dataset
  query_id    = honeycombio_query.error_count.id
  name        = "Errors"
  description = "Count of errors"
}

resource "honeycombio_query" "avg_latency" {
  dataset = var.dataset

  query_json = jsonencode({
    calculations = [{ op = "AVG", column = "duration_ms" }]
    filters = [
      { column = "mcp.method.name", op = "=", value = "tools/call" }
    ]
    time_range = 3600
  })
}

resource "honeycombio_query_annotation" "avg_latency" {
  dataset     = var.dataset
  query_id    = honeycombio_query.avg_latency.id
  name        = "Avg Latency"
  description = "Average tool call latency"
}

resource "honeycombio_query" "p95_latency" {
  dataset = var.dataset

  query_json = jsonencode({
    calculations = [{ op = "P95", column = "duration_ms" }]
    filters = [
      { column = "mcp.method.name", op = "=", value = "tools/call" }
    ]
    time_range = 3600
  })
}

resource "honeycombio_query_annotation" "p95_latency" {
  dataset     = var.dataset
  query_id    = honeycombio_query.p95_latency.id
  name        = "P95 Latency"
  description = "95th percentile tool call latency"
}

resource "honeycombio_query" "tool_call_by_tool" {
  dataset = var.dataset

  query_json = jsonencode({
    calculations = [{ op = "COUNT" }]
    breakdowns   = ["gen_ai.tool.name"]
    filters = [
      { column = "mcp.method.name", op = "=", value = "tools/call" }
    ]
    time_range = 3600
  })
}

resource "honeycombio_query_annotation" "tool_call_by_tool" {
  dataset     = var.dataset
  query_id    = honeycombio_query.tool_call_by_tool.id
  name        = "Tool Calls by Tool"
  description = "Tool call distribution"
}

resource "honeycombio_query" "latency_by_tool" {
  dataset = var.dataset

  query_json = jsonencode({
    calculations = [
      { op = "P50", column = "duration_ms" },
      { op = "P95", column = "duration_ms" }
    ]
    breakdowns = ["gen_ai.tool.name"]
    filters = [
      { column = "mcp.method.name", op = "=", value = "tools/call" }
    ]
    time_range = 3600
  })
}

resource "honeycombio_query_annotation" "latency_by_tool" {
  dataset     = var.dataset
  query_id    = honeycombio_query.latency_by_tool.id
  name        = "Latency by Tool (P50, P95)"
  description = "Tool latency percentiles"
}

resource "honeycombio_query" "errors_by_tool" {
  dataset = var.dataset

  query_json = jsonencode({
    calculations = [{ op = "COUNT" }]
    breakdowns   = ["gen_ai.tool.name"]
    filters = [
      { column = "status_code", op = "=", value = "STATUS_CODE_ERROR" }
    ]
    time_range = 3600
  })
}

resource "honeycombio_query_annotation" "errors_by_tool" {
  dataset     = var.dataset
  query_id    = honeycombio_query.errors_by_tool.id
  name        = "Errors by Tool"
  description = "Error distribution by tool"
}

resource "honeycombio_query" "calls_by_service" {
  dataset = var.dataset

  query_json = jsonencode({
    calculations = [{ op = "COUNT" }]
    breakdowns   = ["service.component"]
    time_range   = 3600
  })
}

resource "honeycombio_query_annotation" "calls_by_service" {
  dataset     = var.dataset
  query_id    = honeycombio_query.calls_by_service.id
  name        = "Calls by Component"
  description = "Call distribution by service component"
}

# ============================================================================
# Outputs
# ============================================================================

output "board_url" {
  description = "URL to the ACCESS-CI Observability board"
  value       = honeycombio_flexible_board.access_observability.board_url
}
