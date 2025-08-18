import pandas as pd
import streamlit as st
from great_tables import GT, style, loc

# This has been added so that Pandas will infer the string type correctly,
# rather than returning it as the "object" data type.
# It will become standard when Pandas 3.0 is released.
pd.options.future.infer_string = True

##### Landing page configuration #####

st.set_page_config(
        page_title="Code Club Streamlit Demonstration", page_icon="📈",
        layout='wide'
    )


# Fix the width of the sidebar using HTML

st.markdown(
    """
    <style>
    /* Adjust the sidebar width */
    [data-testid="stSidebar"] {
        min-width: 200px; /* Set your desired width */
        max-width: 200px;
    }
    </style>
    """,
    unsafe_allow_html=True # this is required to be able to use custom HTML and CSS in the app
)

##### Landing page contents #####

st.title("Code Club Streamlit Demonstration")

st.markdown('''
            In its [own words](https://streamlit.io/), Streamlit is an open-source app framework
            that can be installed like any other Python library and can be used to turn data
            scripts into shareable web apps in minutes, all in Python, without any front-end
            development experience needed.

            This particular app has been developed to showcase some of the features available in
            Streamlit for creating interactive web-based data dashboards.

            **Use the menu on the left-hand side to explore the capabilities of a Streamlit-based
            dashboard.**

            In order to use this app, you will need to upload a dummy data file below. You can
            contact the [Specialist Analytics Team](mailto:scwcsu.analytics.specialist@nhs.net)
            for the files used in the Code Club demonstration, or you can create your own using
            the schema below. (Note that "MODELLING_SERVICE" is a grouping of the "SERVICE" categories).
'''
)

# Create a table to hold the schema
schema = pd.DataFrame(
    {
    'PROVIDER': 'str',
    'SERVICE': 'str',
    'MODELLING_SERVICE': 'str',
    'REFERRAL_DATE': 'datetime64[ns]',
    'REFERRAL_SOURCE': 'str',
    'PRIORITY_TYPE': 'str',
    'REFERRALS_RECEIVED': 'int64',
    'REFERRAL_REASON': 'str'
},
index=[0] # A dataframe needs an index. There's only a single row so we can use 0
)


# Create a function that will validate the columns and data types of the uploaded data
# against the schema.

def validate_df(df, schema = schema):

    # check whether all columns present / no extras
    expected_columns = schema.columns.to_list()
    missing_columns = [col for col in expected_columns if col not in df.columns]
    extra_columns = [col for col in df.columns if col not in expected_columns]

    if missing_columns:
        raise ValueError(f'Columns missing from your dataset: {missing_columns}')
    if extra_columns:
        raise ValueError(f'Unexpected columns in your dataset: {extra_columns}')
    
    # check whether columns are of the correct type
    schema_dict = schema.iloc[0].to_dict()
    df_dict = dict(zip(df.columns.to_list(),df.dtypes.astype(str).tolist()))

    for key, value in schema_dict.items():
        if df_dict[key] != value:
            raise ValueError(f' The data type for column {key} is incorrect. Please refer to the schema above.')

    return True

# Create a function to handle missing and missing-like columns, and also make sure any missing integer column
# gets cast as an integer, but all the others get cast as strings.

def treat_nan(df, default_dtype = 'str', referrals_dtype = 'int'):
    integer_col = 'REFERRALS_RECEIVED'
    missing_like = ['N/A']
    
    for col in df.columns:
        if (df[col].isin(missing_like) | df[col].isna()).all(): # if the column contains missing_like or NaN
            if df[col].name == integer_col:
                df[col] = df[col].astype(referrals_dtype)
            else:
                df[col] = df[col].astype(default_dtype)
    return df


# Create a great_tables object for a better-looking table than the pandas default
# This is what will be rendered on the page.
table = (
    GT(
        schema
    )
    .fmt_markdown()
    .cols_align(align="center")
    .opt_align_table_header(align="center")
    .tab_style(
    style=style.text(weight='normal'),
    locations=loc.column_labels()
    )
)

# Take the HTML output of the great_tables object and render it as a streamlit object.
st.html(table.as_raw_html(), width='stretch')

# File uploader
uploaded_file = st.file_uploader('Upload your .csv file here', type=['csv'])
if uploaded_file is not None:

    uploaded_dataframe = pd.read_csv(uploaded_file,parse_dates=['REFERRAL_DATE'], dayfirst=True)

    treat_nan(uploaded_dataframe)

    if validate_df(uploaded_dataframe) == True:
        st.markdown('**A preview of your data**')
        st.write(uploaded_dataframe.head(10))

        # Add uploaded data to the session state so that it can be re-used on other pages.
        st.session_state['uploaded'] = uploaded_dataframe
    else:
        st.info('Please upload a dataset that matches the schema above')       

else:
    st.info('Please upload a .csv file of your data to continue')
