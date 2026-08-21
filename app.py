import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

from database import (
    create_database,
    get_properties,
    add_property,
    update_property,
    delete_property,
    add_lead,
    get_leads,
    update_lead_status
)
# =========================================================
# LOGIN SYSTEM
# =========================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:

    st.title("Real Estate AI")

    st.subheader("Login")

    username = st.text_input(
        "Username"
    )

    password = st.text_input(
        "Password",
        type="password"
    )

    if st.button(
        "Login",
        use_container_width=True
    ):

        if username == "admin" and password == "admin123":

            st.session_state.logged_in = True
            st.session_state.username = username

            st.success("Login successful!")

            st.rerun()

        else:

            st.error(
                "Invalid username or password."
            )

    st.info(
        "Demo Login: admin / admin123"
    )

    st.stop()

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Real Estate AI",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# DATABASE INITIALIZATION
# =========================================================

create_database()


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f7f9fc;
}

.hero {
    padding: 25px;
    border-radius: 15px;
    background: linear-gradient(135deg, #0f172a, #1e3a8a);
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 38px;
    margin-bottom: 5px;
}

.hero p {
    font-size: 16px;
    opacity: 0.9;
}

.card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    text-align: center;
}

.card-title {
    font-size: 14px;
    color: #64748b;
}

.card-value {
    font-size: 28px;
    font-weight: bold;
    color: #0f172a;
}

.property-card {
    background: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
    margin-bottom: 15px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.04);
}

.property-title {
    font-size: 20px;
    font-weight: bold;
    color: #0f172a;
}

.property-price {
    font-size: 22px;
    font-weight: bold;
    color: #2563eb;
}

.prediction-box {
    background: #eff6ff;
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #bfdbfe;
    text-align: center;
}

.prediction-price {
    font-size: 36px;
    font-weight: bold;
    color: #1d4ed8;
}

.recommendation-box {
    background: #f0fdf4;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #bbf7d0;
    margin-bottom: 15px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD PROPERTIES FROM DATABASE
# =========================================================

df = get_properties()


# =========================================================
# IMPORT OLD CSV DATA IF DATABASE IS EMPTY
# =========================================================

if df.empty:

    try:

        csv_df = pd.read_csv("properties.csv")

        for _, row in csv_df.iterrows():

            add_property(
                row["Location"],
                int(row["BHK"]),
                int(row["Area"]),
                row["Property_Type"],
                float(row["Price"]),
                row["Status"]
            )

        df = get_properties()

    except FileNotFoundError:

        st.error(
            "No properties found. "
            "Please create properties first."
        )

        st.stop()

# =========================================================
# LOAD PROPERTY DATA FROM SQLITE
# =========================================================

create_database()

df = get_properties()


# =========================================================
# IMPORT OLD CSV DATA IF DATABASE IS EMPTY
# =========================================================

if df.empty:

    try:

        csv_df = pd.read_csv("properties.csv")

        for _, row in csv_df.iterrows():

            add_property(
                row["Location"],
                int(row["BHK"]),
                int(row["Area"]),
                row["Property_Type"],
                float(row["Price"]),
                row["Status"]
            )

        df = get_properties()

    except FileNotFoundError:

        st.error(
            "No properties found. "
            "Please create properties first."
        )

        st.stop()

# =========================================================
# DATA VALIDATION
# =========================================================

required_columns = [
    "Location",
    "BHK",
    "Area",
    "Property_Type",
    "Price",
    "Status"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    st.error(
        f"Missing columns: {missing_columns}"
    )

    st.stop()


# =========================================================
# LOCATION MAPPING
# =========================================================

location_map = {
    "Gurgaon": 1,
    "Noida": 2,
    "Delhi": 3,
    "Faridabad": 4
}

for location in df["Location"].unique():

    if location not in location_map:

        location_map[location] = (
            len(location_map) + 1
        )


# =========================================================
# PROPERTY TYPE MAPPING
# =========================================================

property_map = {
    "Apartment": 1,
    "Flat": 2,
    "Villa": 3,
    "House": 4
}

for property_type in df["Property_Type"].unique():

    if property_type not in property_map:

        property_map[property_type] = (
            len(property_map) + 1
        )


# =========================================================
# PREPARE MACHINE LEARNING DATA
# =========================================================

ml_data = df.copy()

ml_data["Location_Code"] = (
    ml_data["Location"].map(location_map)
)

ml_data["Property_Code"] = (
    ml_data["Property_Type"].map(property_map)
)

X = ml_data[
    [
        "Location_Code",
        "BHK",
        "Area",
        "Property_Code"
    ]
]

y = ml_data["Price"]


# =========================================================
# TRAIN RANDOM FOREST MODEL
# =========================================================

model = RandomForestRegressor(
    n_estimators=150,
    random_state=42
)

model.fit(X, y)

# =========================================================
# MODEL EVALUATION
# =========================================================

from sklearn.metrics import mean_absolute_error, r2_score

predictions = model.predict(X)

mae = mean_absolute_error(y, predictions)
r2 = r2_score(y, predictions)

# =========================================================
# FORMAT PRICE
# =========================================================

def format_price(price):

    if price >= 10000000:

        return (
            f"₹{price / 10000000:.2f} Crore"
        )

    return (
        f"₹{price / 100000:.1f} Lakh"
    )


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Real Estate AI")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Property Management",
        "Property Search",
        "Add Property",
        "Price Prediction",
        "AI Recommendations",
        "Lead Management",
        "Analytics",
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "AI-powered real estate property "
    "management and price prediction system."
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Internship Project"
)

# =========================================================
# USER INFO + LOGOUT
# =========================================================

if "username" in st.session_state:

    st.sidebar.success(
        f"Logged in as: {st.session_state.username}"
    )

    if st.sidebar.button(
        "Logout",
        use_container_width=True
    ):

        st.session_state.logged_in = False

        if "username" in st.session_state:
            del st.session_state.username

        st.rerun()

# =========================================================
# MAIN HEADER
# =========================================================

st.markdown("""
<div class="hero">

<h1>Real Estate AI</h1>

<p>
AI-Powered Property Management and Price Prediction System
</p>

</div>
""", unsafe_allow_html=True)


# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    st.subheader("Business Overview")

    total = len(df)

    available = len(
        df[df["Status"] == "Available"]
    )

    sold = len(
        df[df["Status"] == "Sold"]
    )

    average = df["Price"].mean()

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(
            f"""
            <div class="card">

            <div class="card-title">
            Total Properties
            </div>

            <div class="card-value">
            {total}
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            f"""
            <div class="card">

            <div class="card-title">
            Available Properties
            </div>

            <div class="card-value">
            {available}
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            f"""
            <div class="card">

            <div class="card-title">
            Sold Properties
            </div>

            <div class="card-value">
            {sold}
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:

        st.markdown(
            f"""
            <div class="card">

            <div class="card-title">
            Average Property Price
            </div>

            <div class="card-value">
            ₹{average / 10000000:.2f} Cr
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    st.subheader("Quick Market Summary")

    col1, col2, col3 = st.columns(3)

    with col1:

        most_common_location = (
            df["Location"]
            .value_counts()
            .idxmax()
        )

        st.info(
            f"Most Listed Location: "
            f"**{most_common_location}**"
        )

    with col2:

        most_common_type = (
            df["Property_Type"]
            .value_counts()
            .idxmax()
        )

        st.info(
            f"Most Common Property: "
            f"**{most_common_type}**"
        )

    with col3:

        largest_property = df["Area"].max()

        st.info(
            f"Largest Property: "
            f"**{largest_property} sq.ft.**"
        )

    st.markdown("---")

    st.subheader("Featured Available Properties")

    available_df = df[
        df["Status"] == "Available"
    ].head(6)

    if len(available_df) == 0:

        st.warning(
            "No available properties found."
        )

    else:

        for i in range(
            0,
            len(available_df),
            3
        ):

            cols = st.columns(3)

            batch = available_df.iloc[
                i:i + 3
            ]

            for col, (_, property_data) in zip(
                cols,
                batch.iterrows()
            ):

                with col:

                    price_text = format_price(
                        property_data["Price"]
                    )

                    st.markdown(
                        f"""
                        <div class="property-card">

                        <div class="property-title">
                        {property_data["BHK"]}
                        BHK {property_data["Property_Type"]}
                        </div>

                        <p>
                        {property_data["Location"]}
                        </p>

                        <p>
                        {property_data["Area"]} sq.ft.
                        </p>

                        <div class="property-price">
                        {price_text}
                        </div>

                        <p>
                        Available
                        </p>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )
# =========================================================
# PROPERTY MANAGEMENT
# =========================================================

elif page == "Property Management":

    st.subheader("Property Management")

    st.write(
        "Add, edit and delete properties from the system."
    )

    # =====================================================
    # ADD PROPERTY
    # =====================================================

    st.subheader("Add New Property")

    with st.form("add_property_form"):

        col1, col2, col3 = st.columns(3)

        with col1:

            new_location = st.text_input(
                "Location"
            )

            new_bhk = st.number_input(
                "BHK",
                min_value=1,
                max_value=10,
                value=2
            )

        with col2:

            new_area = st.number_input(
                "Area (sq.ft.)",
                min_value=300,
                max_value=20000,
                value=1000,
                step=100
            )

            new_property_type = st.selectbox(
                "Property Type",
                [
                    "Apartment",
                    "Flat",
                    "Villa",
                    "House"
                ]
            )

        with col3:

            new_price = st.number_input(
                "Price (₹)",
                min_value=100000,
                max_value=100000000,
                value=5000000,
                step=100000
            )

            new_status = st.selectbox(
                "Status",
                [
                    "Available",
                    "Sold"
                ]
            )

        add_property_clicked = st.form_submit_button(
            "Add Property",
            use_container_width=True
        )

        if add_property_clicked:

            if new_location.strip() == "":

                st.error(
                    "Location cannot be empty."
                )

            else:

                add_property(
                    new_location.strip(),
                    new_bhk,
                    new_area,
                    new_property_type,
                    new_price,
                    new_status
                )

                st.success(
                    "Property added successfully!"
                )

                st.rerun()

    st.markdown("---")

    # =====================================================
    # EXISTING PROPERTIES
    # =====================================================

    st.subheader("Existing Properties")

    properties = get_properties()

    if properties.empty:

        st.info(
            "No properties found. "
            "Add your first property above."
        )

    else:

        st.dataframe(
            properties,
            use_container_width=True,
            hide_index=True
        )

        # =================================================
        # EDIT PROPERTY
        # =================================================

        st.markdown("---")

        st.subheader("Edit Property")

        property_ids = properties["ID"].tolist()

        selected_id = st.selectbox(
            "Select Property ID",
            property_ids,
            key="edit_property_id"
        )

        selected_property = properties[
            properties["ID"] == selected_id
        ].iloc[0]

        col1, col2, col3 = st.columns(3)

        with col1:

            edit_location = st.text_input(
                "Location",
                value=str(
                    selected_property["Location"]
                ),
                key="edit_location"
            )

            edit_bhk = st.number_input(
                "BHK",
                min_value=1,
                max_value=10,
                value=int(
                    selected_property["BHK"]
                ),
                key="edit_bhk"
            )

        with col2:

            edit_area = st.number_input(
                "Area (sq.ft.)",
                min_value=300,
                max_value=20000,
                value=int(
                    selected_property["Area"]
                ),
                step=100,
                key="edit_area"
            )

            property_types = [
                "Apartment",
                "Flat",
                "Villa",
                "House"
            ]

            current_type = selected_property[
                "Property_Type"
            ]

            edit_type = st.selectbox(
                "Property Type",
                property_types,
                index=(
                    property_types.index(current_type)
                    if current_type in property_types
                    else 0
                ),
                key="edit_type"
            )

        with col3:

            edit_price = st.number_input(
                "Price (₹)",
                min_value=100000,
                max_value=100000000,
                value=int(
                    selected_property["Price"]
                ),
                step=100000,
                key="edit_price"
            )

            edit_status = st.selectbox(
                "Status",
                [
                    "Available",
                    "Sold"
                ],
                index=(
                    0
                    if selected_property["Status"]
                    == "Available"
                    else 1
                ),
                key="edit_status"
            )

        if st.button(
            "Save Changes",
            use_container_width=True
        ):

            if edit_location.strip() == "":

                st.error(
                    "Location cannot be empty."
                )

            else:

                update_property(
                    selected_id,
                    edit_location.strip(),
                    edit_bhk,
                    edit_area,
                    edit_type,
                    edit_price,
                    edit_status
                )

                st.success(
                    "Property updated successfully!"
                )

                st.rerun()

        # =================================================
        # DELETE PROPERTY
        # =================================================

        st.markdown("---")

        st.subheader("Delete Property")

        confirm_delete = st.checkbox(
            "I understand that this property will be permanently deleted.",
            key="confirm_delete"
        )

        if st.button(
            "Delete Selected Property",
            use_container_width=True
        ):

            if not confirm_delete:

                st.error(
                    "Please confirm deletion first."
                )

            else:

                delete_property(
                    selected_id
                )

                st.success(
                    "Property deleted successfully!"
                )

                st.rerun()
# =========================================================
# PROPERTY MANAGEMENT
# =========================================================

elif page == "Property Management":

    st.subheader("Property Management")

    st.write(
        "Add, edit and delete properties from the real estate database."
    )

    # =====================================================
    # ADD PROPERTY
    # =====================================================

    st.markdown("### Add New Property")

    with st.form("add_property_form"):

        col1, col2, col3 = st.columns(3)

        with col1:

            new_location = st.text_input(
                "Location",
                placeholder="Example: Gurgaon"
            )

            new_bhk = st.number_input(
                "BHK",
                min_value=1,
                max_value=10,
                value=2
            )

        with col2:

            new_area = st.number_input(
                "Area (sq.ft.)",
                min_value=300,
                max_value=20000,
                value=1000,
                step=100
            )

            new_property_type = st.selectbox(
                "Property Type",
                [
                    "Apartment",
                    "Flat",
                    "Villa",
                    "House"
                ]
            )

        with col3:

            new_price = st.number_input(
                "Price (₹)",
                min_value=100000,
                max_value=100000000,
                value=5000000,
                step=100000
            )

            new_status = st.selectbox(
                "Status",
                [
                    "Available",
                    "Sold"
                ]
            )

        add_property_button = st.form_submit_button(
            "Add Property",
            use_container_width=True
        )

        if add_property_button:

            if new_location.strip() == "":

                st.error(
                    "Please enter a location."
                )

            else:

                add_property(
                    new_location.strip(),
                    new_bhk,
                    new_area,
                    new_property_type,
                    new_price,
                    new_status
                )

                st.success(
                    "Property added successfully!"
                )

                st.rerun()

    st.markdown("---")

    # =====================================================
    # ALL PROPERTIES
    # =====================================================

    st.subheader("All Properties")

    properties = get_properties()

    if properties.empty:

        st.info(
            "No properties found in the database."
        )

    else:

        st.dataframe(
            properties,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")

        
        # =================================================
        # EDIT PROPERTY
        # =================================================

        st.subheader("Edit Property")

        property_ids = properties["ID"].tolist()

        selected_id = st.selectbox(
            "Select Property ID",
            property_ids
        )

        selected_property = properties[
            properties["ID"] == selected_id
        ].iloc[0]

        col1, col2, col3 = st.columns(3)

        with col1:

            edit_location = st.text_input(
                "Location",
                value=str(
                    selected_property["Location"]
                ),
                key="edit_location"
            )

            edit_bhk = st.number_input(
                "BHK",
                min_value=1,
                max_value=10,
                value=int(
                    selected_property["BHK"]
                ),
                key="edit_bhk"
            )

        with col2:

            edit_area = st.number_input(
                "Area (sq.ft.)",
                min_value=300,
                max_value=20000,
                value=int(
                    selected_property["Area"]
                ),
                step=100,
                key="edit_area"
            )

            edit_type = st.selectbox(
                "Property Type",
                [
                    "Apartment",
                    "Flat",
                    "Villa",
                    "House"
                ],
                index=[
                    "Apartment",
                    "Flat",
                    "Villa",
                    "House"
                ].index(
                    selected_property["Property_Type"]
                )
                if selected_property["Property_Type"]
                in [
                    "Apartment",
                    "Flat",
                    "Villa",
                    "House"
                ]
                else 0,
                key="edit_type"
            )

        with col3:

            edit_price = st.number_input(
                "Price (₹)",
                min_value=100000,
                max_value=100000000,
                value=int(
                    selected_property["Price"]
                ),
                step=100000,
                key="edit_price"
            )

            edit_status = st.selectbox(
                "Status",
                [
                    "Available",
                    "Sold"
                ],
                index=(
                    0
                    if selected_property["Status"]
                    == "Available"
                    else 1
                ),
                key="edit_status"
            )

        if st.button(
            "Save Changes",
            use_container_width=True
        ):

            if edit_location.strip() == "":

                st.error(
                    "Location cannot be empty."
                )

            else:

                update_property(
                    selected_id,
                    edit_location.strip(),
                    edit_bhk,
                    edit_area,
                    edit_type,
                    edit_price,
                    edit_status
                )

                st.success(
                    "Property updated successfully!"
                )

                st.rerun()

        st.markdown("---")

        # =================================================
        # DELETE PROPERTY
        # =================================================

        st.subheader("Delete Property")

        st.warning(
            "Deleting a property permanently removes it "
            "from the database."
        )

        delete_confirm = st.checkbox(
            "I understand that this property will be permanently deleted."
        )

        if st.button(
            "Delete Selected Property",
            use_container_width=True
        ):

            if not delete_confirm:

                st.error(
                    "Please confirm deletion first."
                )

            else:

                delete_property(
                    selected_id
                )

                st.success(
                    "Property deleted successfully!"
                )

                st.rerun()

# =========================================================
# PROPERTY SEARCH
# =========================================================

elif page == "Property Search":

    st.subheader("Find Your Ideal Property")

    col1, col2, col3 = st.columns(3)

    with col1:

        location = st.selectbox(
            "Location",
            ["All"] + sorted(
                df["Location"].unique()
            )
        )

    with col2:

        bhk = st.selectbox(
            "BHK",
            ["All"] + sorted(
                df["BHK"].unique()
            )
        )

    with col3:

        property_type = st.selectbox(
            "Property Type",
            ["All"] + sorted(
                df["Property_Type"].unique()
            )
        )

    status = st.selectbox(
        "Property Status",
        ["All", "Available", "Sold"]
    )

    filtered = df.copy()

    if location != "All":

        filtered = filtered[
            filtered["Location"] == location
        ]

    if bhk != "All":

        filtered = filtered[
            filtered["BHK"] == bhk
        ]

    if property_type != "All":

        filtered = filtered[
            filtered["Property_Type"]
            == property_type
        ]

    if status != "All":

        filtered = filtered[
            filtered["Status"] == status
        ]

    st.markdown("---")

    st.write(
        f"### {len(filtered)} Properties Found"
    )

    if len(filtered) == 0:

        st.warning(
            "No properties match your filters."
        )

    else:

        for _, property_data in filtered.iterrows():

            price_text = format_price(
                property_data["Price"]
            )

            status_text = (
                "Available"
                if property_data["Status"]
                == "Available"
                else "Sold"
            )

            st.markdown(
                f"""
                <div class="property-card">

                <div class="property-title">
                {property_data["BHK"]}
                BHK {property_data["Property_Type"]}
                </div>

                <p>
                <b>Location:</b>
                {property_data["Location"]}
                </p>

                <p>
                <b>Area:</b>
                {property_data["Area"]} sq.ft.
                </p>

                <p>
                <b>Price:</b>

                <span class="property-price">
                {price_text}
                </span>

                </p>

                <p>
                <b>Status:</b>
                {status_text}
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )


# =========================================================
# ADD PROPERTY
# =========================================================

elif page == "Add Property":

    st.subheader("Add New Property")

    st.write(
        "Add a new property to the real estate database."
    )

    st.markdown("---")

    with st.form("property_form"):

        col1, col2 = st.columns(2)

        with col1:

            new_location = st.text_input(
                "Location",
                placeholder="Example: Gurgaon"
            )

            new_bhk = st.number_input(
                "BHK",
                min_value=1,
                max_value=10,
                value=2
            )

            new_area = st.number_input(
                "Area (sq.ft.)",
                min_value=300,
                max_value=10000,
                value=1200,
                step=100
            )

        with col2:

            new_property_type = st.selectbox(
                "Property Type",
                [
                    "Apartment",
                    "Flat",
                    "Villa",
                    "House"
                ]
            )

            new_price = st.number_input(
                "Price (₹)",
                min_value=100000,
                max_value=100000000,
                value=5000000,
                step=100000
            )

            new_status = st.selectbox(
                "Status",
                [
                    "Available",
                    "Sold"
                ]
            )

        add_property_button = st.form_submit_button(
            "Add Property",
            use_container_width=True
        )

    if add_property_button:

        if new_location.strip() == "":

            st.error(
                "Please enter a location."
            )

        else:

            add_property(
                new_location.strip(),
                new_bhk,
                new_area,
                new_property_type,
                new_price,
                new_status
            )

            st.success(
                "Property added successfully!"
            )

            st.rerun()


# =========================================================
# AI PRICE PREDICTION
# =========================================================

elif page == "Price Prediction":

    st.subheader(
        "AI Property Price Prediction"
    )

    st.write(
        "Enter property details and the Machine "
        "Learning model will estimate the market price."
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        prediction_location = st.selectbox(
            "Location",
            sorted(location_map.keys())
        )

        prediction_bhk = st.number_input(
            "Number of BHK",
            min_value=1,
            max_value=10,
            value=3
        )

    with col2:

        prediction_area = st.number_input(
            "Area (sq.ft.)",
            min_value=300,
            max_value=10000,
            value=1500,
            step=100
        )

        prediction_type = st.selectbox(
            "Property Type",
            sorted(property_map.keys())
        )

    st.markdown("---")

    if st.button(
        "Predict Property Price",
        use_container_width=True
    ):

        input_data = pd.DataFrame(
            [[
                location_map[
                    prediction_location
                ],
                prediction_bhk,
                prediction_area,
                property_map[
                    prediction_type
                ]
            ]],
            columns=[
                "Location_Code",
                "BHK",
                "Area",
                "Property_Code"
            ]
        )

        predicted_price = model.predict(
            input_data
        )[0]

        price_text = format_price(
            predicted_price
        )

        st.markdown(
            f"""
            <div class="prediction-box">

            <h3>
            AI Estimated Market Price
            </h3>

            <div class="prediction-price">
            {price_text}
            </div>

            <p>
            Based on location, property size,
            BHK and property type.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.info(
            "This is an ML-based estimate using "
            "the available dataset."
        )
        

# =========================================================
# AI RECOMMENDATIONS
# =========================================================

elif page == "AI Recommendations":

    st.subheader(
        "AI Property Recommendation"
    )

    st.write(
        "Enter your requirements and the system "
        "will recommend suitable properties."
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        preferred_location = st.selectbox(
            "Preferred Location",
            ["Any"] + sorted(
                df["Location"].unique()
            )
        )

        preferred_bhk = st.selectbox(
            "Preferred BHK",
            ["Any"] + sorted(
                df["BHK"].unique()
            )
        )

    with col2:

        budget = st.number_input(
            "Maximum Budget (₹ Lakh)",
            min_value=20,
            max_value=5000,
            value=100,
            step=5
        )

        preferred_type = st.selectbox(
            "Property Type",
            ["Any"] + sorted(
                df["Property_Type"].unique()
            )
        )

    st.markdown("---")

    if st.button(
        "Find Best Properties",
        use_container_width=True
    ):

        recommendation_df = df[
            df["Status"] == "Available"
        ].copy()

        max_budget = budget * 100000

        recommendation_df["Score"] = 0.0

        if preferred_location != "Any":

            recommendation_df.loc[
                recommendation_df["Location"]
                == preferred_location,
                "Score"
            ] += 40

        else:

            recommendation_df["Score"] += 40

        if preferred_bhk != "Any":

            recommendation_df.loc[
                recommendation_df["BHK"]
                == preferred_bhk,
                "Score"
            ] += 25

        else:

            recommendation_df["Score"] += 25

        if preferred_type != "Any":

            recommendation_df.loc[
                recommendation_df["Property_Type"]
                == preferred_type,
                "Score"
            ] += 20

        else:

            recommendation_df["Score"] += 20

        recommendation_df["Budget_Difference"] = abs(
            recommendation_df["Price"]
            - max_budget
        )

        recommendation_df.loc[
            recommendation_df["Price"]
            <= max_budget,
            "Score"
        ] += 15

        recommendation_df = (
            recommendation_df
            .sort_values(
                by=[
                    "Score",
                    "Budget_Difference"
                ],
                ascending=[
                    False,
                    True
                ]
            )
            .head(5)
        )

        if len(recommendation_df) == 0:

            st.warning(
                "No suitable properties found."
            )

        else:

            st.success(
                f"{len(recommendation_df)} "
                "properties recommended!"
            )

            for _, property_data in (
                recommendation_df.iterrows()
            ):

                price_text = format_price(
                    property_data["Price"]
                )

                score = min(
                    int(property_data["Score"]),
                    100
                )

                st.markdown(
                    f"""
                    <div class="recommendation-box">

                    <div class="property-title">
                    {property_data["BHK"]}
                    BHK {property_data["Property_Type"]}
                    </div>

                    <p>
                    <b>Location:</b>
                    {property_data["Location"]}
                    </p>

                    <p>
                    <b>Area:</b>
                    {property_data["Area"]} sq.ft.
                    </p>

                    <p>
                    <b>Price:</b>
                    <span class="property-price">
                    {price_text}
                    </span>
                    </p>

                    <p>
                    <b>AI Match Score:</b>
                    {score}%
                    </p>

                    <p>
                    Available
                    </p>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


# =========================================================
# LEAD MANAGEMENT
# =========================================================

elif page == "Lead Management":

    st.subheader("Customer Lead Management")

    st.write(
        "Manage customer enquiries and track the sales process "
        "from initial enquiry to final closure."
    )

    # -----------------------------------------------------
    # LOAD LEADS FROM DATABASE
    # -----------------------------------------------------

    leads = get_leads()

    # -----------------------------------------------------
    # LEAD STATISTICS
    # -----------------------------------------------------

    total_leads = len(leads)

    new_leads = len(
        leads[leads["Status"] == "New"]
    )

    site_visits = len(
        leads[leads["Status"] == "Site Visit"]
    )

    closed_leads = len(
        leads[leads["Status"] == "Closed"]
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Leads",
            total_leads
        )

    with col2:
        st.metric(
            "New Leads",
            new_leads
        )

    with col3:
        st.metric(
            "Site Visits",
            site_visits
        )

    with col4:
        st.metric(
            "Closed",
            closed_leads
        )

    st.markdown("---")

    # -----------------------------------------------------
    # ADD NEW LEAD
    # -----------------------------------------------------

    st.subheader("Add Customer Enquiry")

    with st.form("lead_form"):

        col1, col2 = st.columns(2)

        with col1:

            customer_name = st.text_input(
                "Customer Name"
            )

            phone = st.text_input(
                "Phone Number"
            )

            budget = st.number_input(
                "Budget (₹ Lakh)",
                min_value=1.0,
                max_value=1000.0,
                value=100.0
            )

        with col2:

            property_options = [
                f'{row["BHK"]} BHK '
                f'{row["Property_Type"]} - '
                f'{row["Location"]} - '
                f'{format_price(row["Price"])}'
                for _, row in df.iterrows()
            ]

            if property_options:

                interested_property = st.selectbox(
                    "Interested Property",
                    property_options
                )

            else:

                interested_property = ""

                st.warning(
                    "No properties available."
                )

            lead_status = st.selectbox(
                "Lead Status",
                [
                    "New",
                    "Contacted",
                    "Site Visit",
                    "Negotiation",
                    "Closed"
                ]
            )

        submitted = st.form_submit_button(
            "Add Lead",
            use_container_width=True
        )

        if submitted:

            if customer_name.strip() == "":

                st.error(
                    "Please enter the customer name."
                )

            elif phone.strip() == "":

                st.error(
                    "Please enter the phone number."
                )

            else:

                add_lead(
                    customer_name.strip(),
                    phone.strip(),
                    interested_property,
                    budget,
                    lead_status
                )

                st.success(
                    "Customer enquiry added successfully!"
                )

                st.rerun()

    st.markdown("---")

    # -----------------------------------------------------
    # DISPLAY LEADS
    # -----------------------------------------------------

    st.subheader("Customer Enquiries")

    leads = get_leads()

    if len(leads) == 0:

        st.info(
            "No customer leads yet. "
            "Add your first enquiry above."
        )

    else:

        st.dataframe(
            leads,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")

        # -------------------------------------------------
        # UPDATE LEAD STATUS
        # -------------------------------------------------

        st.subheader("Update Lead Status")

        lead_ids = leads["ID"].tolist()

        selected_lead_id = st.selectbox(
            "Select Lead",
            lead_ids
        )

        selected_status = st.selectbox(
            "New Status",
            [
                "New",
                "Contacted",
                "Site Visit",
                "Negotiation",
                "Closed"
            ]
        )

        if st.button(
            "Update Status",
            use_container_width=True
        ):

            update_lead_status(
                selected_lead_id,
                selected_status
            )

            st.success(
                f"Lead #{selected_lead_id} "
                f"updated to '{selected_status}'."
            )

            st.rerun()


# =========================================================
# ANALYTICS
# =========================================================

elif page == "Analytics":

    st.subheader(
        "Real Estate Market Analytics"
    )

    col1, col2 = st.columns(2)

    with col1:

        location_prices = (
            df.groupby("Location")["Price"]
            .mean()
            .reset_index()
        )

        location_prices[
            "Average Price (Cr)"
        ] = (
            location_prices["Price"]
            / 10000000
        )

        fig1 = px.bar(
            location_prices,
            x="Location",
            y="Average Price (Cr)",
            title="Average Property Price by Location",
            text_auto=".2f"
        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )

    with col2:

        property_counts = (
            df["Property_Type"]
            .value_counts()
            .reset_index()
        )

        property_counts.columns = [
            "Property Type",
            "Count"
        ]

        fig2 = px.pie(
            property_counts,
            names="Property Type",
            values="Count",
            title="Property Distribution"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    st.markdown("---")

    st.subheader(
        "Area vs Property Price"
    )

    fig3 = px.scatter(
        df,
        x="Area",
        y="Price",
        color="Location",
        size="BHK",
        hover_data=[
            "Property_Type",
            "Status"
        ],
        title="Property Area vs Price"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader(
        "Available vs Sold Properties"
    )

    status_counts = (
        df["Status"]
        .value_counts()
        .reset_index()
    )

    status_counts.columns = [
        "Status",
        "Count"
    ]

    fig4 = px.bar(
        status_counts,
        x="Status",
        y="Count",
        title="Property Status Distribution",
        text_auto=True
    )

    st.plotly_chart(
        fig4,
        use_container_width=True
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "AI-Powered Real Estate Property Management and "
    "Price Prediction System | Internship Project | "
    "Developed by Ved Yadav"
)