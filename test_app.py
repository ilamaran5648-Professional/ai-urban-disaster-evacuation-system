import unittest
import json
import os
from app import build, D, FEATURES, level

class TestSystemSystem(unittest.TestCase):

    def setUp(self):
        self.rain_levels = [0.5, 1.0, 1.5, 2.0]
        self.thr_levels = [0.4, 0.7, 0.95]

    def test_evacuation_requirement_conservation(self):
        """Verify assigned + unplaced == evac for every zone under all storm levels."""
        for rain in self.rain_levels:
            for thr in self.thr_levels:
                res = build(mult=rain, thr=thr, manual=set())
                for z in res["zones"]:
                    evac = z["evac"]
                    assigned = z["assigned"]
                    unplaced = z["unplaced"]
                    self.assertEqual(
                        assigned + unplaced,
                        evac,
                        f"Zone {z['id']} under rain={rain}, thr={thr}: assigned ({assigned}) + unplaced ({unplaced}) != evac ({evac})"
                    )

    def test_shelter_capacity_limits(self):
        """Verify allocated people never exceed shelter physical capacity."""
        for rain in self.rain_levels:
            for thr in self.thr_levels:
                res = build(mult=rain, thr=thr, manual=set())
                for s in res["shelters"]:
                    used = s["used"]
                    cap = s["cap"]
                    self.assertLessEqual(
                        used,
                        cap,
                        f"Shelter {s['id']} used ({used}) exceeds physical capacity ({cap}) under rain={rain}"
                    )

    def test_flooded_shelters_exclusion_and_free_capacity(self):
        """Verify flooded shelters receive 0 allocations and report 0 safe available capacity."""
        for rain in self.rain_levels:
            for thr in self.thr_levels:
                res = build(mult=rain, thr=thr, manual=set())
                for s in res["shelters"]:
                    if not s["safe"]:
                        self.assertEqual(
                            s["used"],
                            0,
                            f"Flooded shelter {s['id']} received {s['used']} allocated evacuees under rain={rain}"
                        )
                        self.assertEqual(
                            s["free"],
                            0,
                            f"Flooded shelter {s['id']} reported free capacity {s['free']} instead of 0"
                        )
                        self.assertEqual(
                            s["safe_cap"],
                            0,
                            f"Flooded shelter {s['id']} reported safe_cap {s['safe_cap']} instead of 0"
                        )

    def test_summary_consistency(self):
        """Verify global summary totals match individual zone and shelter sums."""
        for rain in self.rain_levels:
            res = build(mult=rain, thr=0.7, manual=set())
            s = res["summary"]
            self.assertEqual(
                s["evacuees"],
                s["assigned"] + s["unplaced"],
                f"Summary evacuees ({s['evacuees']}) != assigned ({s['assigned']}) + unplaced ({s['unplaced']})"
            )
            total_zone_evac = sum(z["evac"] for z in res["zones"])
            total_zone_assigned = sum(z["assigned"] for z in res["zones"])
            total_zone_unplaced = sum(z["unplaced"] for z in res["zones"])
            self.assertEqual(s["evacuees"], total_zone_evac)
            self.assertEqual(s["assigned"], total_zone_assigned)
            self.assertEqual(s["unplaced"], total_zone_unplaced)

    def test_physical_capacity_preservation(self):
        """Verify physical shelter capacities match original dataset and are never artificially increased."""
        original_caps = {s["id"]: s["cap"] for s in D["shelters"]}
        res = build(mult=1.0, thr=0.7, manual=set())
        for s in res["shelters"]:
            self.assertEqual(
                s["cap"],
                original_caps[s["id"]],
                f"Shelter {s['id']} physical capacity changed from {original_caps[s['id']]} to {s['cap']}"
            )

    def test_road_classification_counts(self):
        """Verify open_roads + manually_blocked + flood_blocked == total roads."""
        for rain in self.rain_levels:
            manual_set = {"R1", "R2"} if rain > 1.0 else set()
            res = build(mult=rain, thr=0.7, manual=manual_set)
            s = res["summary"]
            total_roads = len(res["roads"])
            self.assertEqual(
                s["open_roads"] + s["manually_blocked"] + s["flood_blocked"],
                total_roads,
                "Sum of open, manually blocked, and flood-blocked roads does not equal total roads"
            )
            self.assertEqual(
                s["manually_blocked"] + s["flood_blocked"],
                s["blocked"],
                "Sum of manually blocked and flood-blocked roads does not equal summary['blocked']"
            )

    def test_alternative_route_validity_and_distinctness(self):
        """Verify alternative routes use only open roads, target safe shelters, and differ from recommended routes."""
        res = build(mult=1.0, thr=0.7, manual=set())
        road_lookup = {r["id"]: r for r in res["roads"]}
        shelter_lookup = {s["id"]: s for s in res["shelters"]}

        has_at_least_one_alt = False
        for zid, rdata in res["routes"].items():
            rec = rdata.get("recommended")
            alt = rdata.get("alternative")
            self.assertIsNotNone(rec, f"Zone {zid} missing recommended route")
            if alt is not None:
                has_at_least_one_alt = True
                # 1. Alternative route must be distinct
                self.assertNotEqual(
                    alt["roads"],
                    rec["roads"],
                    f"Zone {zid} alternative route is identical to recommended route"
                )
                # 2. Alternative route must use open roads only
                for rid in alt["roads"]:
                    self.assertFalse(
                        road_lookup[rid]["blocked"],
                        f"Zone {zid} alternative route uses blocked road {rid}"
                    )
                # 3. Alternative route target shelter must be safe
                dest_shelter = shelter_lookup[alt["shelter"]]
                self.assertTrue(
                    dest_shelter["safe"],
                    f"Zone {zid} alternative route leads to flooded shelter {alt['shelter']}"
                )

        self.assertTrue(has_at_least_one_alt, "Expected at least one zone to have a valid alternative route")

    def test_no_route_when_all_roads_blocked(self):
        """Verify that when all roads leaving a zone are manually blocked, no route is generated and clear notes are logged."""
        # Zone Z1 (Riverside) connects via R1, R2, R3, R14, R25, R31
        all_z1_roads = {"R1", "R2", "R3", "R14", "R25", "R31"}
        res = build(mult=1.5, thr=0.7, manual=all_z1_roads)
        self.assertNotIn("Z1", res["routes"], "Z1 should have no valid route when all outgoing roads are blocked")
        z1_name = next(z["name"] for z in res["zones"] if z["id"] == "Z1")
        note_found = any(f"{z1_name}: no open path" in n for n in res["notes"])
        self.assertTrue(note_found, f"Expected note indicating {z1_name} has no open path to safe shelter")

    def test_ml_transparency_metadata_and_metrics(self):
        """Verify model_info dictionary provides synthetic data labels, holdout R2, feature importances, and limitations."""
        res = build(mult=1.0, thr=0.7, manual=set())
        mi = res.get("model_info")
        self.assertIsNotNone(mi, "model_info missing from backend API response")
        self.assertIn("Synthetic", mi["data_source"], "Data source must clearly label synthetic origin")
        self.assertIsNotNone(mi["holdout_r2"], "Holdout R2 metric must be reported")
        self.assertGreaterEqual(mi["holdout_r2"], 0.9, "Holdout R2 should reflect high surrogate model fit on holdout set")
        self.assertIn("limitation_disclaimer", mi, "Limitation disclaimer must be included")
        self.assertEqual(set(mi["feature_importance"].keys()), set(FEATURES), "Feature importances must match model input features")

    def test_dashboard_backend_summary_fields(self):
        """Verify backend summary provides all KPI fields required by front-end dashboard."""
        res = build(mult=1.0, thr=0.7, manual=set())
        summary_keys = {"high", "medium", "low", "evacuees", "assigned", "unplaced", "total_cap", "total_safe_cap", "free_beds", "open_roads", "manually_blocked", "flood_blocked", "blocked", "isolated"}
        self.assertTrue(summary_keys.issubset(set(res["summary"].keys())), f"Missing summary fields: {summary_keys - set(res['summary'].keys())}")

    def test_risk_level_classification_wording(self):
        """Verify level mapping returns Low, Medium, High for risk scores."""
        self.assertEqual(level(0.20), "Low")
        self.assertEqual(level(0.50), "Medium")
        self.assertEqual(level(0.80), "High")

if __name__ == "__main__":
    unittest.main()
