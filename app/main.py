# import os
# import sys

# # Get the current working directory
# current_dir = os.getcwd()

# # Add the models directory to sys.path to use custom functions
# sys.path.append(os.path.abspath(os.path.join(current_dir, 'models')))

import datetime
import json
import streamlit as st
from datetime import datetime as d

from predict_nohops import predict_nohops_flight_fare
from predict_nohops_return import predict_nohops_return_flight_fare, encode_cyclical_features
from predict_withhops import predict_neural_network


###############################################################################
#
#   CUSTOM FUNCTIONS
#
###############################################################################

def print_trip_summary(trips: list):
    for trip in trips:
        print(trip, end="\n")


###############################################################################
#
#   PROPERTIES
#
###############################################################################

# Import distance data from dataset
with open("models/distance_data.json", "r") as f:
    distance_data = json.load(f)

# Import airport names and IATA codes mapping
with open("models/names_data.json", "r") as f:
    names_data = json.load(f)

# Session stateS
if "disabled" not in st.session_state:
    st.session_state["disabled"] = True
if "number_of_stops" not in st.session_state:
    st.session_state["number_of_stops"] = None

# Parse full airport names and IATA codes
airport_names = [f"{names_data[k]} ({k})" for k in names_data.keys()]

# Define today's date for all tabs
todays_date = datetime.datetime.today().date()

# Journey summary
journey_summary = f"""
## Your journey summary is below:
"""

# Welcome text
welcome = """*:rainbow[Your local estimator for all things airfare!]*

Have you ever wondered if you'd been overcharged airfare by the wealthy 
capitalist airlines? Do you desire knowing about the fairest possible fares
before booking your tickets?

**Look no further!** Our airfare predictor will help you do exactly that, 
using the powers of our shiny in-house Machine Learning algorithms. 

Go tell your airlines that you've been overcharged, and you would like a refund! 

Go tell your friends and family about this app too!

"""

# Spinner message
spinner_msg = "On it, just a bit more..."

# Error messages
error_same_orig_dest = "💀 Origin and destination cannot be the same!"
error_mc = "💀 The first trip's origin and the final trip's destination cannot be the same in multi-city mode."

###############################################################################
#
#   MAIN UI
#
###############################################################################

st.title("🧙 Airfare Wizard 🔮")

st.write(welcome)

st.success(
    """👇 Select your preferences below to get started. """
)

# Define tabs
tab_one_way, tab_return, tab_multicity = st.tabs(["One way", "Return", "Multi-city"])


###############################################################################
#
#   TAB 1: ONE WAY
#
###############################################################################

with tab_one_way:
    st.header("Predict a one-way ticket")
    ow_orig_dest_cols = st.columns(2)
    with ow_orig_dest_cols[0]:
        ow_origin_airport = st.selectbox("From", airport_names, index=0)
    with ow_orig_dest_cols[1]:
        ow_destination_airport = st.selectbox("To", airport_names, index=1)

    if ow_destination_airport == ow_origin_airport:
        st.error(error_same_orig_dest)

    ow_date_time_cols = st.columns(2)
    with ow_date_time_cols[0]:
        ow_dte = st.date_input(f'Date', 
                    min_value=todays_date, 
                    max_value=todays_date + datetime.timedelta(days=365), 
                    format="YYYY/MM/DD")
    with ow_date_time_cols[1]:
        ow_tme = st.time_input("Time", datetime.time(10, 0))

    ow_cabin_cols = st.columns(2)
    with ow_cabin_cols[0]:
        ow_basic_econ = st.toggle("Basic economy only?", help="This is the fare type. If you choose basic economy, then you can only choose the 'Coach' cabin in the next section. Otherwise, you can choose any cabin.")
        if not ow_basic_econ:
            with ow_cabin_cols[1]:
                ow_cabin = st.selectbox("Cabin type", ("Coach", "Premium coach", "Business", "First"))
        else:
            with ow_cabin_cols[1]:
                ow_cabin = st.selectbox(
                    "Cabin type", ("Coach"),
                    help="If you choose 'Basic economy only', you can only book a 'coach' cabin.",          
                    disabled=st.session_state.disabled)
                ow_cabin = "Coach"
    
     # Trip summary
    st.write("--------")
    st.write(journey_summary)
    summary_container = st.container(border=True)
    summary_container.write(f"**From:** {ow_origin_airport}")
    summary_container.write(f"**To:** {ow_destination_airport}")
    summary_container.write(f"**Date:** {ow_dte}")
    summary_container.write(f"**Time:** {ow_tme}")
    summary_container.write(f"**Basic economy:** {'Yes' if ow_basic_econ else 'No'}")
    summary_container.write(f"**Cabin:** {ow_cabin}")
    summary_container.write("\n")

    if st.button("Predict!", key="predict_one_way"):
        if ow_destination_airport == ow_origin_airport:
            st.error(error_same_orig_dest)
        else:
            # Proceed with prediction only if multicity is not selected
            with st.spinner(spinner_msg):
                # Prepare input data for the prediction function
                input_date = ow_dte.strftime('%Y-%m-%d')
                input_time = ow_tme.strftime('%H:%M')
                ow_predicted_fare = predict_nohops_flight_fare(
                    input_date, input_time, ow_origin_airport, 
                    ow_destination_airport, ow_cabin.lower())
                st.write(f'Predicted fare for one-way trip: **:green[${ow_predicted_fare:.2f}]**')


###############################################################################
#
#   TAB 2: RETURN
#
###############################################################################

with tab_return:
    st.header("Predict a return ticket")
    rt_orig_dest_cols = st.columns(2)
    with rt_orig_dest_cols[0]:
        rt_origin_airport = st.selectbox("From ", airport_names, index=0)
    with rt_orig_dest_cols[1]:
        rt_destination_airport = st.selectbox("To ", airport_names, index=1)
    if rt_destination_airport == rt_origin_airport:
        st.error(error_same_orig_dest)

    rt_date_time_cols = st.columns(3)
    with rt_date_time_cols[0]:
        rt_dtes = st.date_input(
            f'Dates', 
            value=(todays_date, todays_date+datetime.timedelta(days=6)),
            min_value=todays_date,
            max_value=todays_date + datetime.timedelta(days=365),
            format="YYYY/MM/DD")
    with rt_date_time_cols[1]:
        rt_dep_tme = st.time_input("Time (departing flight)", datetime.time(10, 0))
    with rt_date_time_cols[2]:
        rt_ret_tme = st.time_input("Time (returning flight)", datetime.time(10, 0))

    rt_cabin_cols = st.columns(3)
    with rt_cabin_cols[0]:
        rt_basic_econ = st.toggle(
            "Basic economy only? ", 
            help="This is the fare type. If you choose basic economy, then you can only choose the 'Coach' cabin in the next section. Otherwise, you can choose any cabin.")
        if not rt_basic_econ:
            with rt_cabin_cols[1]:
                rt_dep_cabin = st.selectbox("Cabin (departing flight)", ("Coach", "Premium coach", "Business", "First"))
            with rt_cabin_cols[2]:
                rt_ret_cabin = st.selectbox("Cabin (returning flight)", ("Coach", "Premium coach", "Business", "First"))
        else:
            with rt_cabin_cols[1]:
                rt_dep_cabin = st.selectbox("Cabin (departing flight)", ("Coach"),
                                        help="If you choose 'Basic economy only', you can only book a 'coach' cabin.",          
                                        disabled=st.session_state.disabled)
            with rt_cabin_cols[2]:
                rt_ret_cabin = st.selectbox("Cabin (returning flight)", ("Coach"),
                                        help="If you choose 'Basic economy only', you can only book a 'coach' cabin.",          
                                        disabled=st.session_state.disabled)
            rt_dep_cabin = "Coach"
            rt_ret_cabin = "Coach"
    
    # Trip summary
    st.write("--------")
    st.write(journey_summary)
    summary_container = st.container(border=True)
    summary_container.write(f"**From:** {rt_origin_airport}")
    summary_container.write(f"**To:** {rt_destination_airport}")
    if len(rt_dtes) == 2:
        summary_container.write(f"**Dates:** {rt_dtes[0]} - {rt_dtes[1]}")
    else:
        summary_container.write(f"**Dates:** {rt_dtes[0]} - ")
    summary_container.write(f"**Basic economy:** {'Yes' if rt_basic_econ else 'No'}")
    summary_container.write(f"**Time (departing flight):** {rt_dep_tme}")
    summary_container.write(f"**Time (returning flight):** {rt_ret_tme}")
    summary_container.write(f"**Cabin (departing flight):** {rt_dep_cabin}")
    summary_container.write(f"**Cabin (returning flight):** {rt_ret_cabin}")
    summary_container.write("\n")
    
    if st.button("Predict!", key="predict_return"):
        if rt_destination_airport == rt_origin_airport:
            st.error(error_same_orig_dest)
        else:
            # Proceed with prediction only if multicity is not selected
            with st.spinner(spinner_msg):
                # Prepare input data for the prediction function
                dep_input_date = rt_dtes[0].strftime('%Y-%m-%d')
                dep_input_time = rt_dep_tme.strftime('%H:%M')
                ret_input_date = rt_dtes[1].strftime('%Y-%m-%d')
                ret_input_time = rt_ret_tme.strftime('%H:%M')
                rt_predicted_dep_fare = predict_nohops_return_flight_fare(
                    dep_input_date, dep_input_time, rt_origin_airport, 
                    rt_destination_airport, rt_dep_cabin.lower())
                rt_predicted_ret_fare = predict_nohops_return_flight_fare(
                    ret_input_date, ret_input_time, rt_destination_airport, 
                    rt_origin_airport, rt_ret_cabin.lower())
                st.write(f'Predicted fare for departing trip: **:green[${rt_predicted_dep_fare:.2f}]**')
                st.write(f'Predicted fare for returning trip: **:green[${rt_predicted_ret_fare:.2f}]**')


###############################################################################
#
#   TAB 3: MULTI-CITY
#
###############################################################################
with tab_multicity:
    mc_origin_at_each_hop = []
    mc_dest_at_each_hop = []
    mc_depart_dates_at_each_hop = []
    mc_depart_times_at_each_hop = []
    mc_cabins_at_each_hop = []

    st.header("Predict a multi-city ticket")
    n_hops_cols = st.columns(3)
    with n_hops_cols[0]:
        n_hops = st.number_input("**Number of trips (max. 4)**", min_value=2, max_value=4)
    mc_basic_econ = st.toggle(
        f"Basic economy only? (applies to all trips)", key=f"mc_basic_econ", 
        help="This is the fare type. If you choose basic economy, then you can only choose the 'Coach' cabin in the next section. Otherwise, you can choose any cabin.")
    for h in range(1, n_hops+1):
        st.subheader(f"Trip {h}:")

        # Column 1: Origin and destination
        orig_dest_cols = st.columns(2)
        with orig_dest_cols[0]:
            mc_origin_airport = st.selectbox(
                f"From", airport_names, index=h-1, key=f"mc_origin_trip_{h}")
        with orig_dest_cols[1]:
            mc_destination_airport = st.selectbox(
                f"To", airport_names, index=h, key=f"mc_destination_trip_{h}")
        if mc_destination_airport == mc_origin_airport:
            st.error(error_same_orig_dest)
        # Append data for summary
        mc_origin_at_each_hop.append(mc_origin_airport)
        mc_dest_at_each_hop.append(mc_destination_airport)

        # Column 2: Date and time
        date_time_cols = st.columns(3)
        with date_time_cols[0]:
            mc_dte = st.date_input(
                f"Date", min_value=todays_date, 
                max_value=todays_date + datetime.timedelta(days=365), 
                format="YYYY/MM/DD", key=f"mc_date_trip_{h}")
            # Take the first departure date only, for fare prediction
            if h == 1:
                mc_depart_date = mc_dte
            mc_depart_dates_at_each_hop.append(mc_dte)
        with date_time_cols[1]:
            mc_tme = st.time_input(
                f"Time", datetime.time(10, 0), 
                key=f"mc_time_trip_{h}")
            mc_depart_times_at_each_hop.append(mc_tme)
        with date_time_cols[2]:
            if not mc_basic_econ:
                mc_cabin = st.selectbox(
                    f"Cabin type", ("Coach", "Premium coach", "Business", "First"),
                    key=f"mc_cabin_trip_{h}")
            else: 
                mc_cabin = st.selectbox(
                    f"Cabin type", ("Coach"),
                    help="If you choose 'Basic economy only', you can only book a 'coach' cabin.",
                    key=f"mc_cabin_trip_{h}_disabled",
                    disabled=st.session_state.disabled)
                mc_cabin = "Coach"
            # Append data for prediction
            mc_cabins_at_each_hop.append(mc_cabin)
    
    # Trip summary
    st.write("--------")
    st.write(journey_summary)
    summary_container = st.container(border=True)
    for i in range(len(mc_origin_at_each_hop)):
        summary_container.subheader(f"**Trip {i+1}:**")
        summary_container.write(f"**From:** {mc_origin_at_each_hop[i]}")
        summary_container.write(f"**To:** {mc_dest_at_each_hop[i]}")
        summary_container.write(f"**Date:** {mc_depart_dates_at_each_hop[i]}")
        summary_container.write(f"**Time:** {mc_depart_times_at_each_hop[i]}")
        summary_container.write(f"**Basic economy:** {'Yes' if mc_basic_econ else 'No'}")
        summary_container.write(f"**Cabin:** {mc_cabins_at_each_hop[i]}")
        summary_container.write("\n")
    
    if st.button("Predict!", key="predict_multicity"):
        trip_datetimes_validator = [
                d.combine(mc_depart_dates_at_each_hop[i], mc_depart_times_at_each_hop[i]) < d.combine(mc_depart_dates_at_each_hop[i-1], mc_depart_times_at_each_hop[i-1]) 
                for i in range(1, len(mc_depart_dates_at_each_hop))
            ]
        if mc_origin_at_each_hop[0] == mc_dest_at_each_hop[-1]:
            st.error(error_mc)
        elif any(trip_datetimes_validator):
            st.error(f"💀 Departure date/time of **Trip {trip_datetimes_validator.index(True)+2}** needs to be after that of **Trip {trip_datetimes_validator.index(True)+1}**.")
        else:
            with st.spinner(spinner_msg):
                    predicted_fare = float(predict_neural_network(
                    origin=mc_origin_at_each_hop[0],
                    dest=mc_dest_at_each_hop[-1], 
                    search_date=todays_date,
                    depart_date=mc_depart_dates_at_each_hop[0], 
                    depart_time=mc_depart_times_at_each_hop[0],
                    is_basic_econ=mc_basic_econ,
                    n_hops=n_hops,
                    cabins=[i.lower() for i in mc_cabins_at_each_hop]).item())
            st.write(f'Predicted fare for multi-city trip: **:green[${predicted_fare:.2f}]**')
