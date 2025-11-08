import json


def load_data(path="data/cloud_storage.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def retrieve_info(parsed):
    """
    根据 LLM 解析出的条件从 JSON 里查信息
    """
    data = load_data()
    platform = parsed.get("Platform")
    query = parsed.get("Query")

    if not platform:
        print("[DEBUG] no platform detected")
        return None

    # 找出对应平台
    target = next((p for p in data if p["Platform"] == platform), None)
    if not target:
        print(f"[DEBUG] platform {platform} not found in JSON")
        return None

    print(f"[STEP 2] found platform {platform} with {len(target['Plans'])} plans")

    # 示例：只处理 cheapest
    if query in ("cheapest", "lowest"):
        cheapest = min(
            (
                (float(opt["Price"]), plan, opt["PlanType"])
                for plan in target["Plans"]
                for opt in plan["PricingOptions"]
                if opt["Price"]
            ),
            key=lambda x: x[0],
        )
        price, plan, plan_type = cheapest
        result = {
            "Platform": platform,
            "PlanName": plan["PlanName"],
            "Price": price,
            "PlanType": plan_type,
        }
        print(f"[STEP 3] retrieved cheapest plan → {result}")
        return result

    print(f"[DEBUG] query type '{query}' not implemented yet")
    return None
