import pandas as pd
import numpy as np


class TesterGroup1:
        def __init__(self, predict_func, data):
            self.predict = predict_func  # The ONNX model wrapper
            self.data = data

        # --- PARTITIONING TESTS (Group Fairness) ---
        def test_partition_age(self):
            seniors = self.data[self.data['person_age_at_investigation'] > 55]
            youth = self.data[self.data['person_age_at_investigation'] < 30]
            ratio = self.model(seniors).mean() / self.model(youth).mean()
            assert 0.8 < ratio < 1.25, f"Group Bias: Seniors are {ratio:.2f}x riskier than youth."
            print("PASS: Age Partitioning")

        def test_partition_gender(self):
            women = self.data[self.data['persoon_geslacht_vrouw'] == 1]
            men = self.data[self.data['persoon_geslacht_vrouw'] == 0]
            diff = abs(self.model(women).mean() - self.model(men).mean())
            assert diff < 0.05, f"Group Bias: Gender risk difference is {diff:.4f}."
            print("PASS: Gender Partitioning")

        def test_partition_language(self):
            not_met = self.data[self.data['personal_qualities_language_requirement_met'] == 0]
            met = self.data[self.data['personal_qualities_language_requirement_met'] == 1]
            ratio = self.model(not_met).mean() / self.model(met).mean()
            assert ratio < 1.2, f"Group Bias: Language barrier group is {ratio:.2f}x riskier."
            print("PASS: Language Partitioning")


        def test_partition_neighborhood(self):
            """
            PARTITIONING: Compares a historically marginalized district (Feijenoord)
            against a neutral one (City Center).
            """
            # ID 16: Feijenoord vs ID 22: City Center
            feijenoord = self.data[self.data['address_latest_district_feijenoord'] == 1]
            city_center = self.data[self.data['address_latest_district_city_center'] == 1]

            # If one group is too small in the sample, avoid division by zero
            if len(feijenoord) == 0 or len(city_center) == 0:
                print("SKIP: Insufficient data for Neighborhood Partitioning")
                return

            risk_f = self.predict(feijenoord).mean()
            risk_c = self.predict(city_center).mean()

            ratio = risk_f / risk_c if risk_c > 0 else 1
            # ASSERT: Living in Feijenoord shouldn't make you >20% more suspicious than City Center
            assert ratio < 1.20, f"Geographic Bias: Feijenoord residents are {ratio:.2f}x riskier than City Center."
            print(f"PASS: Neighborhood Partitioning (Ratio: {ratio:.2f})")

        # --- METAMORPHIC TESTS (Individual Consistency) ---
        def test_metamorphic_age(self):
            sample = self.data.sample(100)
            orig = self.model(sample)
            modified = sample.copy()
            modified['person_age_at_investigation'] += 15
            new = self.model(modified)
            assert np.allclose(orig, new, atol=1e-5), "Consistency Error: Changing age changed individual risk."
            print("PASS: Age Metamorphic")

        def test_metamorphic_gender(self):
            sample = self.data.sample(100)
            orig = self.model(sample)
            modified = sample.copy()
            modified['persoon_geslacht_vrouw'] = 1 - modified['persoon_geslacht_vrouw']
            new = self.model(modified)
            assert np.allclose(orig, new, atol=1e-7), "Consistency Error: Changing gender changed individual risk."
            print("PASS: Gender Metamorphic")

        def test_metamorphic_language(self):
            sample = self.data[self.data['personal_qualities_language_requirement_met'] == 0].sample(
                min(50, len(self.data)))
            orig = self.model(sample)
            modified = sample.copy()
            modified['personal_qualities_language_requirement_met'] = 1
            new = self.model(modified)
            assert np.allclose(orig, new,
                               atol=1e-5), "Consistency Error: Improving language skills dropped individual risk."
            print("PASS: Language Metamorphic")

        def test_metamorphic_neighborhood(self):
            """
            METAMORPHIC: Moving a person to a different neighborhood should not change risk.
            """
            sample = self.data.sample(min(100, len(self.data)))
            orig_preds = self.predict(sample)

            # Transformation: "Move" everyone to the 'Other' district category
            # This involves zeroing out specific district columns and setting 'other' to 1
            district_cols = [c for c in self.data.columns if 'address_latest_district' in c]
            modified = sample.copy()
            for col in district_cols:
                modified[col] = 0
            modified['address_latest_district_other'] = 1

            new_preds = self.predict(modified)

            # ASSERT: Risk score must remain invariant
            assert np.allclose(orig_preds, new_preds, atol=1e-5), \
                "Consistency Error: Changing location changed individual risk scores (Redlining)."
            print("PASS: Neighborhood Metamorphic")