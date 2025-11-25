import streamlit as st
import pickle
import pandas as pd
import base64
import os

MODEL_DIR ="model"

# ========== PAGE CONFIG ==========
st.set_page_config(page_title="Traffic Accident Severity Prediction",
                   layout="wide",
                   page_icon="🚦")

# ========== LOAD MODEL & ENCODERS ==========
@st.cache_resource
def load_stage_models():
    """Cache both stage models so they don't reload every click."""
    with open(os.path.join(MODEL_DIR,"xgb_fatal_notfatal.pkl"), "rb") as f1:
        stage1_model = pickle.load(f1)
    with open(os.path.join(MODEL_DIR,"xgb_serious_slight.pkl"), "rb") as f2:
        stage2_model = pickle.load(f2)
    with open(os.path.join(MODEL_DIR,"encoders.pkl"), "rb") as f3:
        encoders = pickle.load(f3)
    with open(os.path.join(MODEL_DIR,"scaler.pkl"), "rb") as f4:
        scaler = pickle.load(f4)
    return stage1_model, stage2_model, encoders, scaler

stage1_model, stage2_model, encoders, scaler = load_stage_models()


# ========== FEATURE GROUPS ==========
features_page1 = [
    'Day_of_week', 'Age_band_of_driver', 'Sex_of_driver',
    'Educational_level', 'Driving_experience', 'Owner_of_vehicle',
    'Area_accident_occured', 'Lanes_or_Medians', 'Types_of_Junction',
    'Road_surface_type'
]

features_page2 = [
    'Light_conditions', 'Type_of_collision',
    'Number_of_vehicles_involved', 'Number_of_casualties',
    'Vehicle_movement', 'Casualty_class', 'Casualty_severity',
    'Pedestrian_movement', 'Cause_of_accident', 'Time_of_Day'
]

# ========== SIDEBAR NAVIGATION ==========
st.sidebar.markdown("<h2 style='color:white; background-color:#5a5a5a; text-align:center; padding:10px;'>Navigation</h2>", unsafe_allow_html=True)
page = st.sidebar.radio("", ["Home", "About", "Predict"])

# ========== PAGE: HOME ==========
if page == "Home":
    def add_bg_from_local(image_file):
        with open(image_file, "rb") as f:
            data = f.read()
        encoded = base64.b64encode(data).decode()

        st.markdown(
            f"""
            <style>
            [data-testid="stAppViewContainer"] {{
                background: url("data:image/jpeg;base64,{encoded}") no-repeat center center fixed;
                background-size: cover;
            }}
            [data-testid="stHeader"], [data-testid="stToolbar"] {{
                background: rgba(0, 0, 0, 0);
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

    add_bg_from_local("assets/Traffic.jpg")

    # White centered title text
    st.markdown(
        """
        <h1 style='
            color: white;
            text-align: center;
            font-size: 60px;
            font-weight: bold;
            text-transform: uppercase;
            text-shadow: 3px 3px 8px rgba(0,0,0,0.8);
            margin-top: 250px;
        '>
        TRAFFIC ACCIDENT SEVERITY PREDICTION
        </h1>
        """,
        unsafe_allow_html=True
    )

# ========== PAGE: ABOUT ==========
elif page == "About":
    st.markdown(
        """
        <style>
        .stApp {
            background-color: black;
        }
        .banner {
            background-color: maroon;
            padding: 15px;
            text-align: center;
        }
        .banner h1 {
            color: white;
        }
        .content {
            color: white;
            font-size: 18px;
            margin-top: 30px;
            line-height: 1.7;
        }
        .highlight {
            color: #FFB6C1;
            font-weight: bold;
        }
        </style>
        <div class="banner"><h1>ABOUT</h1></div>
        <div class="content">
        <p>
        This web application is designed to <span class="highlight">predict the severity of road traffic accidents</span> 
        based on various contributing factors. The system aims to assist in understanding accident patterns 
        and improving road safety by identifying the likelihood of <span class="highlight">Slight Injury</span>, 
        <span class="highlight">Serious Injury</span>, or <span class="highlight">Fatal Injury</span>.
        </p>

        <h3>Dataset Preparation</h3>
        <p>
        The model was trained using the dataset <span class="highlight">Road.csv</span>, 
        which consisted of <span class="highlight">12,316 records</span> and <span class="highlight">31 attributes</span> 
        representing various accident-related features. The dataset included a target variable categorized into 
        three classes: <b>Slight Injury</b>, <b>Serious Injury</b>, and <b>Fatal Injury</b>. 
        Missing values were carefully handled through imputation, and class imbalance was addressed by 
        applying <span class="highlight"> class weights</span> during model training. 
        All categorical features were <span class="highlight">encoded using Label Encoding</span> to ensure 
        compatibility with the machine learning algorithms.
        Feature selection is done using the SelectFromModel() and 20 features were chosen.
        </p>

        <h3>Model Development</h3>
        <p>
        Multiple models were trained and compared, including 
        <b>K-Nearest Neighbors (KNN)</b>, <b>Gaussian Naive Bayes</b>, <b>Support Vector Classifier (SVC)</b>, 
        <b>Decision Tree</b>, <b>Random Forest</b>, and ensemble methods such as 
        <b>AdaBoost</b>, <b>Gradient Boosting</b>, and <b>XGBoost</b>. 
        Among these, <span class="highlight">XGBoost</span> emerged as the most efficient model, 
        offering strong handling of imbalanced data, faster computation, and superior predictive accuracy. 
        It effectively minimized overfitting while maintaining model interpretability and stability.
        </p>

        <h3>Two-Stage Model Functioning</h3>
        <p>
        The system adopts a <span class="highlight">two-stage classification approach</span>:
        </p>
        <ul>
            <li><b>Stage 1:</b> The model identifies whether an accident is <b>Fatal</b> or <b>Not Fatal</b> 
            based on a threshold value. 
            This stage achieved an <b>accuracy of 83%</b>, successfully identifying fatal cases with a 
            <b>recall of 0.62</b>.</li>
            <li><b>Stage 2:</b> For non-fatal cases, a secondary model classifies the severity into 
            <b>Slight Injury</b> or <b>Serious Injury</b>. 
            This stage achieved an <b>accuracy of 70.6%</b>, with a strong performance in recognizing 
            slight injuries (recall 0.76) and reasonable identification of serious injuries (recall 0.41).</li>
        </ul>
        <p>
        Through this two-tiered approach, the model effectively balances precision and recall, 
        ensuring reliable and interpretable predictions for traffic accident severity analysis.
        </p>

        <p style="margin-top:20px; color:#aaa;">
        <i>Developed as part of a Machine Learning project to enhance predictive road safety insights.</i>
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ========== PAGE: PREDICT ==========
elif page == "Predict":
    st.markdown(
        """
        <style>
        .banner {
            background-color: maroon;
            padding: 15px;
            text-align: center;
        }
        .banner h1 {
            color: white;
        }
        .stApp {
            background-color: white;
        }
        </style>
        <div class="banner"><h1>PREDICT</h1></div>
        """,
        unsafe_allow_html=True
    )

    # Initialize session state
    if "page_num" not in st.session_state:
        st.session_state.page_num = 1
    if "user_inputs" not in st.session_state:
        st.session_state.user_inputs = {}


    def get_options(feature):
        options = {
            'Day_of_week': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],

            'Age_band_of_driver': ['Under 18', '18-30', '31-50', 'Over 51'],

            'Sex_of_driver': ['Male', 'Female'],

            'Educational_level': [
                'Illiterate', 'Writing & reading', 'Elementary school',
                'Junior high school', 'High school', 'Above high school'
            ],

            'Driving_experience': ['No Licence', 'Below 1yr', '1-2yr', '2-5yr', '5-10yr', 'Above 10yr'],

            'Owner_of_vehicle': ['Owner', 'Organization', 'Governmental'],

            'Area_accident_occured': [
                'School areas', 'Church areas', 'Market areas', 'Office areas',
                'Hospital areas', 'Industrial areas', 'Residential areas',
                'Recreational areas', 'Rural village areas', 'Outside rural areas'
            ],

            'Lanes_or_Medians': [
                'One way',
                'Two-way (divided with solid lines road marking)',
                'Two-way (divided with broken lines road marking)',
                'Undivided Two way',
                'Double carriageway (median)'
            ],

            'Types_of_Junction': ['No junction', 'T Shape', 'Y Shape', 'X Shape', 'O Shape', 'Crossing'],

            'Road_surface_type': [
                'Asphalt roads', 'Asphalt roads with some distress',
                'Gravel roads', 'Earth roads'
            ],

            'Light_conditions': ['Daylight', 'Darkness - lights lit', 'Darkness - lights unlit',
                                 'Darkness - no lighting'],

            'Type_of_collision': [
                'Vehicle with vehicle collision', 'Collision with pedestrians',
                'Collision with roadside-parked vehicles', 'Collision with roadside objects',
                'Collision with animals', 'Fall from vehicles', 'Rollover', 'With Train'
            ],

            'Vehicle_movement': [
                'Parked', 'Waiting to go', 'Moving Backward', 'Going straight',
                'Overtaking', 'Reversing', 'Turning left', 'U-Turn',
                'Entering a junction', 'Stopping', 'Turnover', 'Getting off'
            ],

            'Casualty_class': ['Driver or rider', 'Passenger', 'Pedestrian'],

            'Casualty_severity': ['Fatal', 'Serious', 'Slight'],

            'Pedestrian_movement': [
                'Not a Pedestrian',
                'Crossing from driver\'s nearside',
                'Crossing from nearside - masked by parked or stationary vehicle',
                'Crossing from offside - masked by parked or stationary vehicle',
                'Walking along in carriageway, facing traffic',
                'Walking along in carriageway, back to traffic',
                'In carriageway, stationary - not crossing (standing or playing)'
            ],

            'Cause_of_accident': [
                'Driving carelessly', 'Overspeed', 'Overtaking', 'No priority to vehicle',
                'No priority to pedestrian', 'No distancing', 'Improper parking',
                'Changing lane to the left', 'Changing lane to the right',
                'Driving at high speed', 'Driving under the influence of drugs',
                'Drunk driving', 'Moving Backward', 'Overloading',
                'Driving to the left', 'Getting off the vehicle improperly',
                'Overturning', 'Turnover'
            ],

            'Time_of_Day': ['Morning', 'Afternoon', 'Evening', 'Night'],

            'Number_of_vehicles_involved': [],
            'Number_of_casualties': []
        }

        return options.get(feature, [])

    # Collect inputs
    if st.session_state.page_num == 1:
        st.markdown("### Step 1: Enter First 10 Details")
        for feature in features_page1:
            if "Number" in feature:
                value = st.number_input(f"{feature}", min_value=0, step=1)
            else:
                value = st.selectbox(f"{feature}", get_options(feature))
            st.session_state.user_inputs[feature] = value

        if st.button("Next ➡️"):
            st.session_state.page_num = 2
            st.rerun()

    elif st.session_state.page_num == 2:
        st.markdown("### Step 2: Enter Remaining Details")
        for feature in features_page2:
            if "Number" in feature:
                value = st.number_input(f"{feature}", min_value=0, step=1)
            else:
                value = st.selectbox(f"{feature}", get_options(feature))
            st.session_state.user_inputs[feature] = value

        if st.button("🔙 Back"):
            st.session_state.page_num = 1
            st.rerun()

        if st.button("Predict 🚦"):
            input_data = pd.DataFrame([st.session_state.user_inputs])
            for col in input_data.columns:
                if col in encoders:
                    input_data[col] = encoders[col].transform(input_data[col])

            # Scale numerical features
            scaled_data = scaler.transform(input_data)

            fatal_prob = stage1_model.predict_proba(scaled_data)[0][1]
            fatal_threshold = 0.003

            if fatal_prob >= fatal_threshold:
                st.error(f"💀 **Fatal Injury**")
            else:
                # --- Stage 2: Serious vs Slight ---
                stage2_pred = stage2_model.predict(scaled_data)[0]

                if stage2_pred == 1:
                    st.warning(f"⚠️ **Serious Injury**")
                else:
                    st.info(f"🟢 **Slight Injury**")