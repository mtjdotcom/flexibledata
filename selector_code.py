import os
import gspread
import tempfile
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from pandasai.llm import OpenAI
from pandasai import SmartDataframe
from pandasai.responses.response_parser import ResponseParser
from oauth2client.service_account import ServiceAccountCredentials


# title
st.set_page_config(
    page_title="LP Data Analysis App"
)

## Password function ######################################################################
def check_password():

    def password_entered():
        """Checks whether password is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

if check_password():

    # Load the dataset
    @st.cache_data

    def load_data():
        # Define the scope
        secrets = st.secrets["gcp_service_account"]
        scope = ['https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_dict(secrets, scope)

        # Authorize the clientsheet 
        client = gspread.authorize(creds)

        # Open the sheet
        sheet_url = st.secrets["private_gsheets_url"] 
        sheet = client.open_by_url(sheet_url)
        # Get the first sheet of the Spreadsheet
        sheet_instance = sheet.get_worksheet(0)

        # Convert to DataFrame
        data = pd.DataFrame(sheet_instance.get_all_records())

        # Your existing processing
        data['Close Date'] = pd.to_datetime(data['Close Date'])
        data['Close Quarter'] = data['Close Date'].dt.year.astype(str) + " Q" + data['Close Date'].dt.quarter.astype(str)
        data = data.round(1)
        return data

    df = load_data()

    tab1, tab2, tab3 = st.tabs(["LP Data", "AI","Excel"])

    st.sidebar.title("LPA Data Tab")
    st.sidebar.subheader("Pivot and Group your Fund data")

    with tab1:

        # Sidebar: Group by selector
        group_by_column = st.sidebar.selectbox('Index Column (group/pivot):', ['None'] + list(df.columns))

        # Sidebar: I limted the selectors to the object-type columns for filtering
        object_columns = df.select_dtypes(include=['object']).columns
        selected_filters = {}
        for col in object_columns:
            unique_values = df[col].unique().tolist()
            selected_value = st.sidebar.selectbox(f'Filter by {col}:', ['All'] + unique_values, index=0)
            selected_filters[col] = selected_value

        # Apply filters to the DataFrame based on user selection
        filtered_df = df.copy()
        for col, selected_value in selected_filters.items():
            if selected_value != 'All':
                filtered_df = filtered_df[filtered_df[col] == selected_value]

        # Pick a statistical function on the numerical columns after filtering
        numerical_columns = filtered_df.select_dtypes(include=[np.number]).columns
        selected_columns = st.sidebar.multiselect('Numerical Columns:', numerical_columns)

        stats_functions = {
            'Mean': 'mean',
            'Median': 'median',
            'Sum': 'sum',
            'Standard Deviation': 'std',
            'Count': 'count'
        }

        # Dictionary to store selected statistical functions for each column
        selected_stats = {col: st.sidebar.selectbox(f'Statistical function for {col}:', 
                                          list(stats_functions.keys()), key=col) for col in selected_columns}

        # Calculate statistics and group by if selected
        if st.button('Calculate Statistics'):
            if not selected_columns:  # Check if no numerical columns are selected
                st.write("Nothing to calculate, please pick numerical columns")
            else:
                # Prepare the dictionary for aggregation
                agg_funcs = {col: stats_functions[func_name] for col, func_name in selected_stats.items()}
                
                # Group by operation if a column is selected
                if group_by_column != 'None':
                    # Apply the aggregation functions
                    grouped_df = filtered_df.groupby(group_by_column).agg(agg_funcs)
                    # Round the grouped DataFrame to 1 decimal place
                    grouped_df = grouped_df.round(1)
                    st.write(f'Grouped and Statistical Results by {group_by_column}:')
                    st.dataframe(grouped_df)
                else:
                    # If no grouping is selected, just apply the statistical functions directly
                    results = {col: getattr(filtered_df[col].dropna(), func)() for col, func in agg_funcs.items()}
                    results_df = pd.DataFrame([results])
                    # Round the results DataFrame to 1 decimal place
                    results_df = results_df.round(1)
                    st.write('Statistical Results:')
                    st.dataframe(results_df)
        else:
            # Display the filtered DataFrame
            st.caption('Use the slider on the left to select columns to group and to analyze. For example, you can group (aka pivot) your data by Vintage Year (first drop down) and then select numerical columns you wish to show in the table. You can further filter your data by Manager ("Group"), Type, or Geography.')

            st.write('Filtered Data:')

            st.dataframe(filtered_df)


        selected_type = selected_filters.get('Type', 'All')
        st.write("Fund Type: " + selected_type)
                # After displaying the dataframe
        if len(selected_columns) > 0:  # Check if any numerical columns are selected
            # Prepare a summary of the statistical functions applied
            stats_summary = "Statistical Functions Applied:\n"
            for col, stat_function in selected_stats.items():
                stats_summary += f"- {col}: {stat_function}\n"
            
            # Display the summary under the dataframe
            st.write(stats_summary)
        else:
            st.write("No statistical functions were selected for calculation.")
       

    with tab2:
        #using Pandas AI https://pandas-ai.com/
        st.subheader("Chat with your Data")
        st.caption("Try a table: Show me the average seed fund size by vintage year.")
        st.caption("Try a chart: Show me a line chart of the median entry valuation by vintage year. Breakout Seed and Series A funds.")
        st.caption("Please go easy on the API and my wallet! If you experience an error, try asking your request in a different way.")
        with st.expander('Your Data'):
            st.write(df)

        query = st.text_area("🗣️ Chat with your Data (using OpenAI)")
        
        if st.button('Generate Response'):
            if query:
                with st.spinner('Generating response...'):
                    llm = OpenAI(api_token=os.environ['OPENAI_API_KEY'])
                    query_engine = SmartDataframe(df, config={'llm': llm})

                    answer = query_engine.chat(query)
                    if "Unfortunately" in answer:
                         # Use a temporary file for the plot
                        # fig, ax = plt.subplots()
                        image_path = '/mount/src/flexibledata/exports/charts/temp_chart.png'
                        # Display the image
                        st.image(image_path, caption='Your Chart')
                        # st.pyplot()
                        # st.set_option('deprecation.showPyplotGlobalUse', False)
                        # st.pyplot()
                    else:
                        st.write(answer)

    with tab3:
        #Question 1####################################################################################################################################
        def question_1(df, group):
            # Initialize an empty DataFrame
            df_1 = pd.DataFrame()
            
            # Count the total number of funds by 'Vintage Year'
            df_1["Total # of Funds"] = df['Vintage Year'].value_counts()
            
            # Count the number of 'Seed' funds by 'Vintage Year'
            df_1['# of Seed Funds'] = df[df['Type'] == 'Seed'].groupby(group)['Type'].count().astype(int)
            
            # Count the number of 'Series A' funds by 'Vintage Year'
            df_1['# of Series A Funds'] = df[df['Type'] == 'Series A'].groupby(group)['Type'].count().astype(int)
            
            # Calculate the average fund size by 'Vintage Year'
            df_1['Average Fund Size'] = df.groupby(group)['Fund Size ($M)'].mean().round(1)
            
            # Sum of invested capital at the underlying portfolio level by 'Vintage Year'
            df_1['Invested Capital (Underlying Portfolio Level)'] = df.groupby(group)['Total Cost\n($M)'].sum().round(1)
            
            # Sum of total value at the underlying portfolio level by 'Vintage Year'
            df_1['Total Value (Underlying Portfolio Level)'] = df.groupby(group)['Total Value\n($M)'].sum().round(1)
            
            # Calculate the Gross MOC (Multiple of Cost) at the underlying portfolio level per 'Vintage Year'
            df_1['Gross MOC (Underlying Portfolio Level)'] = (df_1['Total Value (Underlying Portfolio Level)'] / df_1['Invested Capital (Underlying Portfolio Level)']).round(1)
            
            # Fill NaN values with 0 for columns that may not have data for all 'Vintage Year'
            df_1.fillna(0, inplace=True)
            
            # Reset index to make 'Vintage Year' a column instead of an index
            df_1.reset_index(inplace=True)
            df_1.rename(columns={'index': group}, inplace=True)
            

            return(df_1)
        
        df_1 = question_1(df, 'Vintage Year')
        st.write('Question 1')
        st.dataframe(df_1)    



        #Question2####################################################################################################################################
        #create TVPI buckets
        st.divider()
        def question_2(df):
            df['TVPI'] = ((df['Fund Distributions ($M)'] + df['Fund FMV ($M)'])/df['Fund \nPaid-In ($M)']).astype(float).round(1)

            bins = [-float('inf'), 1, 3, 5, float('inf')]
            labels = ['0-1x', '1-3x', '3-5x', '5x+']

            # Categorize using cut
            df['Category'] = pd.cut(df['TVPI'], bins=bins, labels=labels, right=False)
            df_2 = pd.DataFrame()
            df_2['TVPI Category'] = df['Category'].value_counts()
            df_2['Percent %'] = ((df_2['TVPI Category']/len(df)) * 100).round(1)

            return(df_2)
        df_2 = question_2(df)
        st.write('Question 2')
        st.table(df_2)

        #question 3####################################################################################################################################
        st.divider()
        sorted_df = df.sort_values(by=['Group', 'Fund'], ascending=[True, True])
        sorted_df['Close Date'] = pd.to_datetime(sorted_df['Close Date'])
        sorted_df['Avg. Fundraising Time Between Funds (Yrs)'] = (sorted_df['Close Date'].diff().dt.days / 365.25).round(1)
        # Condition: 'Fund' column is equal to 1
        condition = sorted_df['Fund'] == 1

        # Set values in 'Fundraising Time Between Funds' to NaN where condition is True
        sorted_df.loc[condition, 'Avg. Fundraising Time Between Funds (Yrs)'] = np.nan

        df_3 = sorted_df.groupby('Vintage Year')['Avg. Fundraising Time Between Funds (Yrs)'].mean().round(1)
        st.write('Question 3')
        st.dataframe(df_3)

        #question 4####################################################################################################################################
        st.divider()
        seed_seriesA_median_valuation = df.groupby(['Vintage Year', 'Type'])['Median Entry Valuation ($M)'].median().unstack()
        # get rid of nan
        seed_seriesA_median_valuation.fillna(0, inplace=True)
        df_4 = seed_seriesA_median_valuation[['Seed', 'Series A']]

        seed_seriesA_median_valuation = df.groupby(['Vintage Year', 'Type'])['Median Entry Valuation ($M)'].median().unstack()
        seed_seriesA_median_valuation.fillna(0, inplace=True)
        df_4 = seed_seriesA_median_valuation[['Seed', 'Series A']]

        st.write('Question 4')
        st.dataframe(df_4)    

        # Resetting index to use 'Vintage Year' as a column for Altair
        df_4 = df_4.reset_index()

        # Melting the DataFrame to long format, which Altair prefers for this type of chart
        df_long = df_4.melt('Vintage Year', var_name='Type', value_name='Median Entry Valuation ($M)')

        # Creating the Altair chart
        chart = alt.Chart(df_long).mark_bar().encode(
            # Combine 'Vintage Year' and 'Type' in the x encoding to place bars side by side
            x=alt.X('Vintage Year:N', title='Vintage Year', axis=alt.Axis(labels=True)),
            y=alt.Y('Median Entry Valuation ($M):Q', title='Median Entry Valuation ($M)'),
            color='Type:N'
        ).properties(
            width=220  # Adjust the width as needed
        )

        # Displaying the chart in Streamlit
        st.altair_chart(chart, use_container_width=True)

        st.divider()

        # Group by 'Vintage Year' and 'Type', then get the median of 'Median Entry Valuation ($M)'
        seed_seriesA_median_valuation = df.groupby(['Vintage Year', 'Type'])['Median Entry Valuation ($M)'].median().unstack()

        # Replace NaN values with 0
        seed_seriesA_median_valuation.fillna(0, inplace=True)

        # Select only the 'Seed' and 'Series A' columns
        df_4 = seed_seriesA_median_valuation[['Seed', 'Series A']]

        # Create a bar chart with 'Vintage Year' on the x-axis and the data next to each other
        fig, ax = plt.subplots()

        # Set the width of the bars
        bar_width = 0.35

        # Set the positions of the bars on the x-axis
        index = np.arange(len(df_4))

        # Plot both 'Seed' and 'Series A' bars
        ax.bar(index, df_4['Seed'], bar_width, label='Seed', edgecolor='none')
        ax.bar(index + bar_width, df_4['Series A'], bar_width, label='Series A')

        # Set the x-axis labels to the 'Vintage Year', which is the index of df_4
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(df_4.index, rotation=90)

        # Adding labels and title
        ax.set_xlabel('Vintage Year')
        ax.set_ylabel('Median Entry Valuation ($M)')
        ax.set_title('Median Entry Valuation by Vintage Year and Type')
        ax.legend()

        st.pyplot(fig)

        st.divider()
        #Question 5################################################################################################################################################
        df_5 = df.groupby('Type')['Commitment ($M)'].mean().round(1)
        st.write('Question 5')
        st.dataframe(df_5)   

        st.divider()
        #Question 6################################################################################################################################################
        port_level = df.copy()
        port_level['% of Fund'] = port_level['Commitment ($M)']/port_level['Fund Size ($M)']
        port_level['FoF Dist'] = port_level['% of Fund'] * port_level['Fund Distributions ($M)']
        port_level['FoF Paid-in'] = port_level['% of Fund'] * port_level['Fund \nPaid-In ($M)']
        port_level['FoF NAV'] = port_level['% of Fund'] * port_level['Total Value\n($M)'] 
        port_level['FoF TVPI'] = (port_level['FoF Dist'] + port_level['FoF NAV'])/port_level['FoF Paid-in']
        df_6 = (port_level['FoF Dist'].sum() + port_level['FoF NAV'].sum()) / port_level['FoF Paid-in'].sum()
        st.write('Question 6')
        st.write(df_6)
