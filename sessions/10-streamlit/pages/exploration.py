import pandas as pd
import streamlit as st
import altair as alt
alt.renderers.enable('default')
import datetime as dt

##### Data exploration page configuration #####

st.set_page_config(
        page_title="Data Exploration", page_icon=":chart_with_upwards_trend:",
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

##### Constants #####
# it's convention to write constants in caps

# Colour schemes for charts
SCHEME_DROPDOWN = alt.param(
            name = 'Colour Scheme',
            bind = alt.binding_select(options=
                                             [
                                            'category20',
                                            'category20b',
                                            'category20c',
                                            'tableau20',
                                            'yellowgreenblue',
                                            'yelloworangered',
                                            'turbo'
                                        ],
                                        name = 'Chart Colour Scheme '
                    ),
                    value='category20'
        )

# Chart width and height

WIDTH = 1000

HEIGHT = 500

##### Data exploration page contents #####

st.title('Data Exploration')

# If it has been uploaded, bring through the data that has been uploaded to the current session.
# Then all of the page contents will be rendered.
# Otherwise, there will be an advisory note to upload some data on the Home page.

if "uploaded" in st.session_state:
    df = st.session_state["uploaded"]

    st.markdown('''
                On this page we can do some exploratory data analysis with the uploaded dataset.
    '''
    )

    tab1, tab2, tab3, tab4 = st.tabs(['Summary', 'Overall Activity', 'Comparison','Modelling Service Volumes'])

    with tab1:

        st.markdown('''
                    ### Data summary and completeness checks
        '''
        )

        st.write(f'Number of rows in dataset: {len(df)}')

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('Percentage of rows with missing values: ')
            missing_ratio = df.isnull().sum()/len(df)
            missing_df = missing_ratio.to_frame(name='Missing')
            missing_df['Missing'] = missing_df['Missing'].apply(lambda x: f'{x:.1%}')
            st.dataframe(missing_df, width=300)

        with col2:
            st.markdown('Summary of the REFERRALS_RECEIVED column: ')
            describe_df = df['REFERRALS_RECEIVED'].describe()
            st.dataframe(describe_df, width=300)

        with col3:
            st.markdown('Unique values by column: ')
            col_select1 = st.selectbox(
                label='Select a column',
                options=set(df.columns) - {'PROVIDER','REFERRAL_DATE','REFERRALS_RECEIVED'}
            )
            selected_df_val_counts = df[col_select1]
            val_counts = selected_df_val_counts.value_counts()
            val_counts_df = val_counts.to_frame(name='Count')
            st.dataframe(val_counts_df,width=500,height=245)
    
    with tab2:

        st.markdown('''
                    ### Overall activity by month
        '''
        )

        df['REFERRAL_MONTH'] = df['REFERRAL_DATE'].dt.to_period('M').dt.to_timestamp()
        df_monthly = df.copy().groupby(['PROVIDER','REFERRAL_MONTH'], as_index=False)[['REFERRALS_RECEIVED']].sum()

        alt_df_monthly = alt.Chart(df_monthly,title='Overall Monthly Activity Count',width=WIDTH, height=HEIGHT).mark_line(point=True).encode(
        x=alt.X('REFERRAL_MONTH:T',axis=alt.Axis(format='%b-%Y',labelAngle=-90,tickCount='month')),
        y='REFERRALS_RECEIVED:Q',
        tooltip=['REFERRALS_RECEIVED']
        ).interactive()
        
        st.altair_chart(alt_df_monthly, use_container_width=False)  

    with tab3:

        st.markdown('''
                    ### Monthly activity by selection
        '''
        )

        col_select2 = st.selectbox(
            label='Select how you would like to group the data',
            options=set(df.columns) - {'PROVIDER','REFERRAL_DATE','REFERRAL_MONTH','REFERRALS_RECEIVED'}
        )

        col_select2_df = df[col_select2]
        
        cat_select1 = st.multiselect(
            label='Select the categories for which you would like to compare the monthly activity',
            options=set(col_select2_df)
        )

        df_selection = (
            df[df[col_select2].isin(cat_select1)].copy()
            .groupby(['PROVIDER','REFERRAL_MONTH',col_select2], as_index=False)[['REFERRALS_RECEIVED']]
            .sum()
        )

        
        alt_df_selection = (
            alt.Chart(df_selection, title='Monthly Activity Count for Selection', width=WIDTH, height=HEIGHT)
            .mark_line(point=True)
            .encode(
                x=alt.X('REFERRAL_MONTH:T',axis=alt.Axis(format='%b-%Y',labelAngle=-90,tickCount='month')),
                y='REFERRALS_RECEIVED:Q',
                color=alt.Color(col_select2).scale(scheme={'expr': 'Colour Scheme'}),
                tooltip=[col_select2,'REFERRALS_RECEIVED']
            )
            .interactive()
            .add_params(
                SCHEME_DROPDOWN
            )
        )
        
        st.altair_chart(alt_df_selection, use_container_width=False)

    with tab4:
        df_modelling_service = (
            df.copy()
            .groupby(['PROVIDER','REFERRAL_MONTH','MODELLING_SERVICE'], as_index=False)[['REFERRALS_RECEIVED']]
            .sum()
        )

        alt_df_modelling = (
            alt.Chart(df_modelling_service, title='Comparison of Monthly Activity per Modelling Service', width=WIDTH, height=HEIGHT)
            .mark_boxplot(extent='min-max')
            .encode(
                x=alt.X('MODELLING_SERVICE:N'),
                y=alt.Y('REFERRALS_RECEIVED:Q'),
                color=alt.Color('MODELLING_SERVICE').scale(scheme={'expr': 'Colour Scheme'}),
                tooltip=['MODELLING_SERVICE','REFERRALS_RECEIVED']
            )
            .interactive()
            .add_params(
                SCHEME_DROPDOWN
            )
        )

        st.altair_chart(alt_df_modelling, use_container_width=False)

else:
    st.info('Please upload a .csv file of your data on the Home page to continue')


