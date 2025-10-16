import sys
import os
try:
    import streamlit as st
except ImportError as e:
    sys.stderr.write(
        "Streamlit is not installed in this Python environment.\n"
        "Install it or use the Python 3.13 interpreter where it is installed.\n\n"
        "Quick fix commands:\n"
        "  python -m pip install streamlit\n"
        "or run with:\n"
        "  \"C:\\Users\\sanka\\AppData\\Local\\Programs\\Python\\Python313\\python.exe\" -m streamlit run app.py\n"
    )
    sys.exit(1)
import pickle
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load Model
model = pickle.load(open("best_random_forest.pkl", "rb"))
label_encoder = joblib.load(r"C:\\Users\\sanka\\Desktop\\new peoject\\label_encoder.pkl")


st.title("🌍 Earthquake Damage Prediction App")
st.markdown("Predict the damage level based on earthquake parameters.")

# Inputs
magnitude = st.slider("Magnitude (Richter Scale)", 0.0, 10.0, 5.5, 0.1)
depth = st.slider("Depth (km)", 0.0, 700.0, 10.0, 1.0)
soil_id = st.number_input("Soil Type (Encoded)", 0, 109, 0)
region_id = st.number_input("Region Cluster", 0, 5, 0)

# Initialize session state to persist results across reruns
if "predicted" not in st.session_state:
    st.session_state.predicted = False
    st.session_state.decoded = None
    st.session_state.probs = None
    st.session_state.class_labels = None
    st.session_state.last_inputs = None

# Debug/Help panel to understand current state
with st.expander("Help & Debug Info", expanded=False):
    st.write({
        "model_type": type(model).__name__,
        "has_predict_proba": hasattr(model, "predict_proba"),
        "has_feature_importances": hasattr(model, "feature_importances_"),
        "last_inputs": st.session_state.get("last_inputs"),
        "predicted": st.session_state.get("predicted"),
    })
    try:
        test_indices = list(range(len(getattr(label_encoder, 'classes_', []))))
        labels_preview = label_encoder.inverse_transform(test_indices) if getattr(label_encoder, 'classes_', None) is not None else []
        st.write({"label_encoder_classes": list(labels_preview)})
    except Exception as e:
        st.write({"label_encoder_error": str(e)})

if st.button("Predict Damage"):
    df = pd.DataFrame([{
        "Magnitude": magnitude,
        "Depth": depth,
        "SoilType_Encoded": soil_id,
        "Region_Cluster": region_id
    }])
    pred = model.predict(df)[0]
    decoded = label_encoder.inverse_transform([pred])[0]

    # Persist results
    st.session_state.predicted = True
    st.session_state.decoded = decoded
    st.session_state.last_inputs = {"Magnitude": magnitude, "Depth": depth, "SoilType_Encoded": soil_id, "Region_Cluster": region_id}

    # Store probabilities (if available)
    probs = None
    class_labels = None
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(df)[0]
            class_indices = list(range(len(probs)))
            try:
                class_labels = label_encoder.inverse_transform(class_indices)
            except Exception:
                class_labels = [str(i) for i in class_indices]
        except Exception as e:
            st.warning(f"Could not compute probabilities: {e}")
    st.session_state.probs = probs.tolist() if probs is not None else None
    st.session_state.class_labels = list(class_labels) if class_labels is not None else None

# Render results outside of the button block so they persist across reruns
if st.session_state.get("predicted", False):
    st.success(f"Predicted Damage Category: {st.session_state.decoded}")

    # Probability distribution bar chart
    if st.session_state.probs is not None and st.session_state.class_labels is not None:
        try:
            probs = np.array(st.session_state.probs)
            class_labels = st.session_state.class_labels
            fig, ax = plt.subplots(figsize=(6, 3.5))
            bars = ax.bar(class_labels, probs, color="#5f6f7f")
            ax.set_ylim(0, 1)
            ax.set_ylabel("Probability")
            ax.set_title("Predicted Class Probabilities")
            for b in bars:
                ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01, f"{b.get_height():.2f}", ha='center', fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not render probability chart: {e}")
    else:
        st.info("Model does not provide probability estimates.")

    # Feature importance bar chart (if available) - show regardless after prediction
    if hasattr(model, "feature_importances_"):
        with st.expander("Feature Importance", expanded=False):
            try:
                feature_names = ["Magnitude", "Depth", "SoilType_Encoded", "Region_Cluster"]
                importances = model.feature_importances_
                order = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)
                ordered_names = [feature_names[i] for i in order]
                ordered_vals = [importances[i] for i in order]
                fig2, ax2 = plt.subplots(figsize=(6, 3.5))
                bars2 = ax2.bar(ordered_names, ordered_vals, color="#6c7a89")
                ax2.set_title("Feature Importance")
                ax2.set_ylabel("Importance")
                for i, v in enumerate(ordered_vals):
                    ax2.text(i, v + (max(ordered_vals) * 0.02), f"{v:.3f}", ha='center', fontsize=8)
                plt.xticks(rotation=15, ha='right')
                plt.tight_layout()
                st.pyplot(fig2)
            except Exception as e:
                st.warning(f"Could not render feature importance: {e}")
    else:
        st.caption("Model does not expose feature_importances_.")

    # Simple scenario exploration (optional) for 'future' magnitudes
    with st.expander("Scenario: Vary Magnitude (Future What-If)", expanded=False):
        steps = st.slider("Number of magnitude steps", 3, 20, 8, 1)
        min_mag, max_mag = st.slider("Magnitude range", 0.0, 10.0, (max(0.0, magnitude-1.0), min(10.0, magnitude+1.0)), 0.1)
        if hasattr(model, 'predict_proba'):
            mags = np.linspace(min_mag, max_mag, steps)
            scenario_df = pd.DataFrame({
                'Magnitude': mags,
                'Depth': depth,
                'SoilType_Encoded': soil_id,
                'Region_Cluster': region_id
            })
            try:
                scenario_probs = model.predict_proba(scenario_df)
                class_indices = list(range(scenario_probs.shape[1]))
                try:
                    class_labels = label_encoder.inverse_transform(class_indices)
                except Exception:
                    class_labels = [str(i) for i in class_indices]
                # Plot stacked probabilities vs magnitude
                fig3, ax3 = plt.subplots(figsize=(6, 4))
                bottom = np.zeros(len(mags))
                for j, cls in enumerate(class_labels):
                    ax3.bar(mags, scenario_probs[:, j], bottom=bottom, width=(mags[1]-mags[0])*0.7 if len(mags)>1 else 0.3, label=cls)
                    bottom += scenario_probs[:, j]
                ax3.set_xlabel('Magnitude')
                ax3.set_ylabel('Probability')
                ax3.set_title('Predicted Class Probability vs Magnitude')
                ax3.set_ylim(0, 1)
                ax3.legend(fontsize=8, ncol=2)
                plt.tight_layout()
                st.pyplot(fig3)
            except Exception as e:
                st.warning(f"Could not compute scenario probabilities: {e}")
        else:
            st.info("Scenario plot unavailable: model has no predict_proba().")

# Guidance when running this file directly
if __name__ == "__main__":
    msg = (
        "This is a Streamlit app. Please run it with one of these commands:\n\n"
        "  streamlit run app.py\n"
        "  python -m streamlit run app.py\n"
        "  \"C:\\Users\\sanka\\AppData\\Local\\Programs\\Python\\Python313\\python.exe\" -m streamlit run app.py\n"
    )
    print(msg)
    # Exit cleanly to avoid nested runtime instances
    sys.exit(0)
