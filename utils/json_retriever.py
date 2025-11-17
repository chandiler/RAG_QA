import json
import re


def load_data(path="data/cloud_storage.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_storage_to_gb(storage_str):
    """
    将 '2 TB' / '100 GB' 转为 GB 数字
    """
    if not storage_str:
        return None

    m = re.match(r"([0-9.]+)\s*(TB|GB)", storage_str, re.I)
    if not m:
        return None

    val = float(m.group(1))
    unit = m.group(2).upper()

    return val * 1024 if unit == "TB" else val


def retrieve_info(parsed):
    data = load_data()

    platform = parsed.get("Platform")
    price_cond = parsed.get("Price", {})
    storage_cond = parsed.get("Storage", {})
    feature = parsed.get("Feature")

    # --- 1) 平台过滤 ---
    if platform:
        platforms = [p for p in data if p["Platform"] == platform]
    else:
        platforms = data[:]  # 搜索所有平台

    candidates = []

    for p in platforms:
        for plan in p["Plans"]:
            # --- 2) storage filtering ---
            st_num = parse_storage_to_gb(plan.get("Storage"))
            if storage_cond:
                if storage_cond.get("min") is not None:
                    if st_num is None or st_num < storage_cond["min"]:
                        continue
                if storage_cond.get("max") is not None:
                    if st_num is None or st_num > storage_cond["max"]:
                        continue

            # --- 3) feature filtering ---
            if feature:
                text = " ".join(plan.get("Features", [])).lower()
                if feature.lower() not in text:
                    continue

            # --- 4) price filtering ---
            valid_pricing = []
            for opt in plan.get("PricingOptions", []):
                price = float(opt["Price"])
                cycle = opt.get("PlanType")

                if price_cond.get("cycle") and cycle != price_cond["cycle"]:
                    continue

                if price_cond.get("min") is not None and price < price_cond["min"]:
                    continue

                if price_cond.get("max") is not None and price > price_cond["max"]:
                    continue

                valid_pricing.append(opt)

            if price_cond and not valid_pricing:
                continue

            # 用匹配到的最便宜 pricing 作为排序依据
            if valid_pricing:
                best = min(valid_pricing, key=lambda x: float(x["Price"]))
            else:
                best = plan["PricingOptions"][0]

            candidates.append((p["Platform"], plan, best))

    if not candidates:
        return None

    # 按价格升序排序
    candidates.sort(key=lambda x: float(x[2]["Price"]))

    platform, plan, opt = candidates[0]
    result = {
        "Platform": platform,
        "PlanName": plan["PlanName"],
        "Price": float(opt["Price"]),
        "PlanType": opt["PlanType"],
        "Storage": plan["Storage"],
        "FeatureMatch": parsed.get("Feature"),
    }

    print(f"[STEP 3] filtered result → {result}")
    return result
