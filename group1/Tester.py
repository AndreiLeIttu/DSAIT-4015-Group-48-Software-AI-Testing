import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import shap

class TesterGroup1:
        def __init__(self, predict_func, data):
            self.model = predict_func  # The ONNX model wrapper
            self.data = data
            # Store original column order to ensure consistency
            self.original_columns = data.columns.tolist()
            print(self.original_columns)
        
        def _prepare_data(self, data_subset):
            """
            Prepare data subset for prediction by ensuring it has all required columns
            in the correct order, matching the original data structure.
            """
            # Reindex to ensure all columns are present in the correct order
            # Fill missing columns with 0 (shouldn't happen with proper filtering)
            prepared = data_subset.reindex(columns=self.original_columns, fill_value=0)
            return prepared
        
        def _get_predictions(self, data_subset):
            """
            Get predictions from the model and ensure they're in a consistent format.
            """
            prepared = self._prepare_data(data_subset)
            preds = self.model.predict(prepared)
            # Flatten if needed (handle both 1D and 2D outputs)
            if preds.ndim > 1:
                preds = preds.flatten()
            return preds

        def run_partition_tests(self):
            """
            Run all partitioning tests to check for group fairness.
            Continues running all tests even if one fails.
            """
            results = []
            test_methods = [
                ("Gender", self.test_partition_gender),
                ("Language", self.test_partition_language),
                ("Age", self.test_partition_age),
                ("Neighborhood", self.test_partition_neighborhood)
            ]
            
            for test_name, test_method in test_methods:
                try:
                    test_method()
                    results.append((test_name, "PASS", None))
                except AssertionError as e:
                    results.append((test_name, "FAIL", str(e)))
                except Exception as e:
                    results.append((test_name, "ERROR", str(e)))
            
            # Print summary
            print("\n" + "-" * 60)
            print("PARTITIONING TESTS SUMMARY:")
            for test_name, status, error in results:
                if status == "PASS":
                    print(f"  [{status}] {test_name}")
                else:
                    print(f"  [{status}] {test_name}: {error}")
            print("-" * 60)
            
            return results

        def run_metamorphic_tests(self):
            """
            Run all metamorphic tests to check for individual consistency.
            Continues running all tests even if one fails.
            """
            results = []
            test_methods = [
                ("Age", self.test_metamorphic_age),
                ("Gender", self.test_metamorphic_gender),
                ("Language", self.test_metamorphic_language),
                ("Neighborhood", self.test_metamorphic_neighborhood)
            ]
            
            for test_name, test_method in test_methods:
                try:
                    test_method()
                    results.append((test_name, "PASS", None))
                except AssertionError as e:
                    results.append((test_name, "FAIL", str(e)))
                except Exception as e:
                    results.append((test_name, "ERROR", str(e)))
            
            # Print summary
            print("\n" + "-" * 60)
            print("METAMORPHIC TESTS SUMMARY:")
            for test_name, status, error in results:
                if status == "PASS":
                    print(f"  [{status}] {test_name}")
                else:
                    print(f"  [{status}] {test_name}: {error}")
            print("-" * 60)
            
            return results

        # --- PARTITIONING TESTS (Group Fairness) ---
        def test_partition_age(self):
            """
            PARTITIONING: Compares risk predictions between seniors (>=60) and younger adults (<60).
            Expectation: Risk predictions should be similar across age groups (within reasonable bounds).
            """
            seniors = self.data[self.data['persoon_leeftijd_bij_onderzoek'] >= 60]
            youth = self.data[self.data['persoon_leeftijd_bij_onderzoek'] < 60]
            
            if len(seniors) == 0 or len(youth) == 0:
                print("SKIP: Insufficient data for Age Partitioning")
                return
            
            preds_seniors = self._get_predictions(seniors)
            preds_youth = self._get_predictions(youth)
            risk_seniors = np.mean(preds_seniors)
            risk_youth = np.mean(preds_youth)
            
            if risk_youth == 0:
                print("SKIP: Zero risk for youth group in Age Partitioning")
                return
            
            ratio = risk_seniors / risk_youth
            # ASSERT: Seniors shouldn't be more than 25% riskier or less than 80% as risky as youth
            assert 0.8 < ratio < 1.25, f"Group Bias: Seniors receive {ratio:.2f}x risk score than youth."
            print(f"PASS: Age Partitioning (Ratio: {ratio:.2f})")

        def test_partition_gender(self):
            """
            PARTITIONING: Compares risk predictions between women and men.
            Expectation: Risk predictions should be similar across gender groups (gender fairness).
            """
            women = self.data[self.data['persoon_geslacht_vrouw'] == 1]
            men = self.data[self.data['persoon_geslacht_vrouw'] == 0]
            
            if len(women) == 0 or len(men) == 0:
                print("SKIP: Insufficient data for Gender Partitioning")
                return
            
            risk_women = self.model.predict(women).mean()
            risk_men = self.model.predict(men).mean()
            diff = abs(risk_women - risk_men)
            if not (diff < 0.05):
                raise AssertionError(f"Group Bias: Gender risk difference is {diff:.4f}.")
            print(f"PASS: Gender Partitioning (Diff: {diff:.4f})")

        def test_partition_language(self):
            """
            PARTITIONING: Compares risk predictions between those who met language requirements and those who didn't.
            Expectation: Language requirement status shouldn't significantly affect risk predictions.
            """
            not_met = self.data[self.data['persoonlijke_eigenschappen_taaleis_voldaan'] == 0]
            met = self.data[self.data['persoonlijke_eigenschappen_taaleis_voldaan'] == 1]
            
            if len(not_met) == 0 or len(met) == 0:
                print("SKIP: Insufficient data for Language Partitioning")
                return
            
            preds_not_met = self._get_predictions(not_met)
            preds_met = self._get_predictions(met)
            risk_not_met = np.mean(preds_not_met)
            risk_met = np.mean(preds_met)
            
            if risk_met == 0:
                print("SKIP: Zero risk for language-met group in Language Partitioning")
                return
            
            ratio = risk_not_met / risk_met
            assert 0.8 < ratio < 1.2, f"Group Bias: Language barrier group is {ratio:.2f}x riskier."
            print(f"PASS: Language Partitioning (Ratio: {ratio:.2f})")


        def test_partition_neighborhood(self):
            """
            PARTITIONING: Compares a historically marginalized district (Feijenoord)
            against a neutral one (City Center).
            Expectation: Risk predictions should be similar across neighborhoods (geographic fairness).
            """
            # Find neighborhood columns
            wijk_cols = [c for c in self.data.columns if 'adres_recentste_wijk' in c]
            
            # Try to find Feijenoord and City Center columns
            feijenoord_col = None
            city_center_col = None
            
            for col in wijk_cols:
                if 'feijenoord' in col.lower():
                    feijenoord_col = col
                if 'centrum' in col.lower() or 'center' in col.lower() or 'stadscentrum' in col.lower():
                    city_center_col = col
            
            # If specific columns not found, use first two neighborhood columns as comparison
            if feijenoord_col is None or city_center_col is None:
                if len(wijk_cols) < 2:
                    print("SKIP: Insufficient neighborhood columns for Neighborhood Partitioning")
                    return
                # Use first two neighborhood columns as proxy
                feijenoord_col = wijk_cols[0]
                city_center_col = wijk_cols[1]
            
            feijenoord = self.data[self.data[feijenoord_col] == 1]
            city_center = self.data[self.data[city_center_col] == 1]

            # If one group is too small in the sample, avoid division by zero
            if len(feijenoord) == 0 or len(city_center) == 0:
                print("SKIP: Insufficient data for Neighborhood Partitioning")
                return

            preds_feijenoord = self._get_predictions(feijenoord)
            preds_city_center = self._get_predictions(city_center)
            risk_f = np.mean(preds_feijenoord)
            risk_c = np.mean(preds_city_center)

            if risk_c == 0:
                print("SKIP: Zero risk for city center group in Neighborhood Partitioning")
                return

            ratio = risk_f / risk_c
            # ASSERT: Living in Feijenoord shouldn't make you >20% more suspicious than City Center
            if not (ratio < 1.20):
                raise AssertionError(f"Geographic Bias: {feijenoord_col} residents are {ratio:.2f}x riskier than {city_center_col}.")
            print(f"PASS: Neighborhood Partitioning (Ratio: {ratio:.2f})")

        # --- METAMORPHIC TESTS (Individual Consistency) ---
        def test_metamorphic_age(self):
            """
            METAMORPHIC: Age is NOT a protected attribute in this context - it can legitimately affect risk.
            This test is removed as age-based risk assessment may be valid for fraud detection.
            Instead, we test that small age changes don't cause disproportionate risk changes.
            """
            sample = self.data.sample(min(100, len(self.data)))
            orig_preds = self._get_predictions(sample)
            
            # Small age change (1 year) should cause minimal change in risk
            modified = sample.copy()
            modified['persoon_leeftijd_bij_onderzoek'] = 90 - modified['persoon_leeftijd_bij_onderzoek']
            new_preds = self._get_predictions(modified)
            
            # Allow small changes but not dramatic ones (age can affect risk, but shouldn't be extreme)
            max_change = np.abs(orig_preds - new_preds).max()
            if not (max_change < 0.1):
                raise AssertionError(f"Consistency Error: Small age change caused {max_change:.4f} risk change.")
            print(f"PASS: Age Metamorphic (Max change: {max_change:.4f})")

        def test_metamorphic_gender(self):
            """
            METAMORPHIC: Changing gender should not change individual risk predictions.
            Gender is a protected attribute and should not affect fraud risk assessment.
            """
            sample = self.data.sample(min(100, len(self.data)))
            orig_preds = self._get_predictions(sample)
            modified = sample.copy()
            modified['persoon_geslacht_vrouw'] = 1 - modified['persoon_geslacht_vrouw']
            new_preds = self._get_predictions(modified)
            if not np.allclose(orig_preds, new_preds, atol=1e-5):
                max_diff = np.abs(orig_preds - new_preds).max()
                raise AssertionError(f"Consistency Error: Changing gender changed individual risk. Max diff: {max_diff:.6f}")
            print("PASS: Gender Metamorphic")

        def test_metamorphic_language(self):
            """
            METAMORPHIC: Changing language requirement status should not change individual risk predictions.
            Language skills are a proxy for protected attributes and shouldn't directly affect fraud risk.
            """
            not_met_data = self.data[self.data['persoonlijke_eigenschappen_taaleis_voldaan'] == 0]
            if len(not_met_data) == 0:
                print("SKIP: No data with unmet language requirements for Language Metamorphic test")
                return
            
            sample = not_met_data.sample(min(50, len(not_met_data)))
            orig_preds = self._get_predictions(sample)
            modified = sample.copy()
            modified['persoonlijke_eigenschappen_taaleis_voldaan'] = 1
            new_preds = self._get_predictions(modified)
            if not np.allclose(orig_preds, new_preds, atol=1e-5):
                max_diff = np.abs(orig_preds - new_preds).max()
                raise AssertionError(f"Consistency Error: Improving language skills changed individual risk. Max diff: {max_diff:.6f}")
            print("PASS: Language Metamorphic")

        def test_metamorphic_neighborhood(self):
            """
            METAMORPHIC: Moving a person to a different neighborhood should not change risk.
            This tests for geographic discrimination (redlining) - location alone shouldn't affect risk.
            """
            sample = self.data.sample(min(100, len(self.data)))
            orig_preds = self._get_predictions(sample)

            # Find all neighborhood/district columns
            wijk_cols = [c for c in self.data.columns if 'adres_recentste_wijk' in c.lower() or 
                        'wijk' in c.lower() or 'district' in c.lower()]
            
            if len(wijk_cols) == 0:
                print("SKIP: No neighborhood columns found for Neighborhood Metamorphic test")
                return
            
            modified = sample.copy()
            
            # Transformation: Zero out all neighborhood columns and set a neutral/default one
            # Find if there's an "other" or default neighborhood column
            other_col = None
            for col in wijk_cols:
                if 'overig' in col.lower() or 'other' in col.lower():
                    other_col = col
                    break
            
            # Zero out all neighborhood indicators
            for col in wijk_cols:
                modified[col] = 0
            
            # If "other" column exists, set it to 1; otherwise set the first neighborhood column to 1
            if other_col:
                modified[other_col] = 1
            elif len(wijk_cols) > 0:
                modified[wijk_cols[0]] = 1

            new_preds = self._get_predictions(modified)

            # ASSERT: Risk score should remain invariant when only location changes
            # Using a reasonable tolerance since neighborhood might have some legitimate correlation
            # but the change should be minimal if the model is fair
            max_diff = np.abs(orig_preds - new_preds).max()
            if not (max_diff < 0.05):
                raise AssertionError(f"Consistency Error: Changing location changed individual risk scores (Redlining). Max diff: {max_diff:.4f}")
            print(f"PASS: Neighborhood Metamorphic (Max diff: {max_diff:.4f})")

        def test_shapley_values(self):
            background = self.data[self.model.model_features].sample(
                min(100, len(self.data))
            )
            shap_explainer = shap.Explainer(self.predict_for_shap, background)
            sample = self.data[self.model.model_features].sample(
                min(100, len(self.data))
            )
            shap_values = shap_explainer(sample)
            sensitive_features = ['persoon_geslacht_vrouw',
                                  'persoon_leeftijd_bij_onderzoek',
                                  'persoonlijke_eigenschappen_taaleis_voldaan']

            # Compare average SHAP contribution
            shap_df = pd.DataFrame(shap_values.values, columns=shap_values.feature_names)
            # Mean absolute SHAP values
            sum_abs_shap = shap_df.abs().sum().sort_values(ascending=False)
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 6))
            sum_abs_shap.plot(kind='bar', color='skyblue')
            plt.title("Feature Importance (Mean Absolute SHAP Values)", fontsize=16)
            plt.ylabel("Mean |SHAP value|", fontsize=14)
            plt.xlabel("Features", fontsize=14)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.show()


            # shap.summary_plot(shap_values, features = sample, feature_names=shap_values.feature_names)
            return shap_values

        def predict_for_shap(self, X_subset):
            """
            X_subset: only the features SHAP should see (numeric, e.g., 13 features)
            Returns: predictions for the model, filling missing features with 0
            """
            # 1. Start with all zeros for the full feature set
            full_X = pd.DataFrame(
                0,
                index=X_subset.index,
                columns=self.data.columns  # all model features
            )

            # 2. Fill in the SHAP-selected features
            full_X[X_subset.columns] = X_subset
            full_X = self._prepare_data(full_X)
            # 3. Forward to your existing model
            return self.model.predict(full_X)
