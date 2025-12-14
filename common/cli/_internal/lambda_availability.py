#!/usr/bin/env python3
"""Check Lambda Labs instance availability by region."""

import os
import sys

import requests

# GPU VRAM mapping (not provided by API, sourced from lambda.ai/instances)
GPU_VRAM_GB = {
    "b200": 180,
    "gh200": 96,
    "h100": 80,
    "h200": 141,
    "a100_80gb": 80,
    "a100": 40,  # default A100 is 40GB unless specified
    "a6000": 48,
    "a10": 24,
    "rtx6000": 24,
    "v100": 16,
}


def get_gpu_memory(instance_type: str, gpu_count: int) -> str:
    """Parse instance type name to determine GPU memory. Returns formatted string like '24GB x 1'."""
    name = instance_type.lower()

    # Check for explicit memory in name (e.g., a100_80gb)
    if "80gb" in name or "80g" in name:
        vram = 80
    elif "40gb" in name or "40g" in name:
        vram = 40
    else:
        # Match GPU model from instance name
        for gpu_model, gb in GPU_VRAM_GB.items():
            if gpu_model in name:
                vram = gb
                break
        else:
            return "-"

    return f"{vram}GB x {gpu_count}"


def get_instance_types(api_key: str) -> dict:
    """Fetch instance types from Lambda API."""
    response = requests.get(
        'https://cloud.lambdalabs.com/api/v1/instance-types',
        auth=(api_key, '')
    )
    return response.json()


def filter_by_region(data: dict, region: str) -> list:
    """Filter instance types available in specified region."""
    available = []

    instance_types = data.get("data", {})

    for instance_name, details in instance_types.items():
        regions_with_capacity = details.get("regions_with_capacity_available", [])

        if any(r.get("name") == region for r in regions_with_capacity):
            available.append({
                "instance_type": instance_name,
                "description": details.get("instance_type", {}).get("description", "N/A"),
                "price": details.get("instance_type", {}).get("price_cents_per_hour", 0) / 100,
                "specs": details.get("instance_type", {}).get("specs", {}),
            })

    return available


def main():
    region = "us-west-1"

    api_key = os.getenv("LAMBDA_API_KEY")
    if not api_key:
        print("Error: LAMBDA_API_KEY environment variable not set")
        print("Usage: export LAMBDA_API_KEY='your-api-key-here'")
        return 1

    print(f"Fetching available instances in {region}...\n")

    data = get_instance_types(api_key)
    available = filter_by_region(data, region)

    if not available:
        print(f"No instances currently available in {region}")
        return 0

    print(f"Available instances in {region}:\n")

    # Prepare table data
    rows = []
    for item in sorted(available, key=lambda x: x["price"]):
        specs = item['specs']
        gpu_count = specs.get('gpus', 0)
        rows.append({
            "type": item['instance_type'],
            "gpus": str(gpu_count) if gpu_count else '-',
            "gpu_mem": get_gpu_memory(item['instance_type'], gpu_count),
            "vcpus": str(specs.get('vcpus', '-')),
            "mem": f"{specs.get('memory_gib', '-')} GiB" if specs.get('memory_gib') else '-',
            "storage": f"{specs.get('storage_gib', '-')} GiB" if specs.get('storage_gib') else '-',
            "price": f"${item['price']:.2f}/hr",
        })

    # Calculate column widths
    headers = ["Price", "Instance Type", "GPU Mem", "GPUs", "vCPUs", "Memory", "Storage"]
    keys = ["price", "type", "gpu_mem", "gpus", "vcpus", "mem", "storage"]
    widths = [max(len(headers[i]), max(len(r[keys[i]]) for r in rows)) for i in range(len(keys))]

    # Print header
    header_row = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(header_row)
    print("-" * len(header_row))

    # Print rows
    for row in rows:
        print("  ".join(row[keys[i]].ljust(widths[i]) for i in range(len(keys))))

    return 0


if __name__ == "__main__":
    sys.exit(main())
