import pandas as pd
import streamlit as st
import altair as alt
import datetime as dt
alt.renderers.enable('default')
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import math
import numpy as np
from sklearn.metrics import mean_squared_error

##### Data exploration page configuration #####

st.set_page_config(
        page_title="Simple Forecasting Demo", page_icon="📈",
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

# Forecast training dataset size

TRAIN_SIZE = 25

##### Forecast demonstration page contents #####

st.title('Simple Forecast Demonstration')

if "uploaded" in st.session_state:
    df = st.session_state["uploaded"]
    df['REFERRAL_MONTH'] = df['REFERRAL_DATE'].dt.to_period('M').dt.to_timestamp()

    df_mod = df.copy().groupby(['PROVIDER','REFERRAL_MONTH','MODELLING_SERVICE'], as_index=False)[['REFERRALS_RECEIVED']].sum()

    tab1,tab2,tab3 = st.tabs(['Background Information','Train-Test','Predictions'])

    with tab1:

        st.markdown('''
                    This demonstration is intended to show how it is possible to have
                    some form of modelling technique set up that can work with data
                    uploaded by the end user. **It should not be viewed as a demonstration
                    of a full forecasting workflow.**

                    It will demonstrate the use of a Holt-Winters forecasting model from the
                    `statsmodels` [package](https://www.statsmodels.org/stable/index.html).
        '''
        )

    with tab2:

        st.header('Training and Testing the Model')

        st.markdown('''
                    Here we can select the type of trend component and kind of seasonality
                    component for our forecasting model, and then see which combination
                    performs best at making a prediction against our test set.

                    We will have the ability to make predictions of activity for each of the
                    modelling services. This is more likely what we would want to forecast, as
                    opposed to the overall monthly activity.

                    In order to have enough data points to train the model, only those modelling
                    services with activity for the whole time period will be made available.
        '''
        )

        ##### Selectboxes for the model options and modelling services #####

        ttcol1, ttcol2, ttcol3 = st.columns([3,1,1]) # the numbers in the square brackets set the relative width

        with ttcol1:
            # create a set of modelling service options where there are at least 30 months' data
            mod_serv_options = {val for val in df_mod['MODELLING_SERVICE'] if len(df_mod[df_mod['MODELLING_SERVICE'] == val]) > 29}
            mod_serv_select = st.selectbox(
                label = 'Modelling Service',
                options= mod_serv_options
            )

        with ttcol2:
            trend = st.selectbox(
            label='Trend',
            options=[None,'additive','multiplicative']
        )
        with ttcol3:
            seasonality = st.selectbox(
                label='Seasonality',
                options=[None,'additive','multiplicative']
            )

        ##### Preparing our model #####
        df_forecast = df_mod[df_mod['MODELLING_SERVICE'] == mod_serv_select].copy()

        df_forecast.set_index('REFERRAL_MONTH',inplace=True)

        # Create train and test datasets
        train_df_forecast = df_forecast['REFERRALS_RECEIVED'].iloc[:TRAIN_SIZE]
        test_df_forecast = df_forecast['REFERRALS_RECEIVED'].iloc[TRAIN_SIZE:]

        # Create model
        hw_train = ExponentialSmoothing(
            train_df_forecast,
            trend=trend,
            seasonal=seasonality,
            seasonal_periods=12 # monthly data; year represents a "season" in terms of the cycle of repetition
        )

        # Make a prediction on the training data alongside the test data
        hw_tt_pred = hw_train.fit().forecast(len(test_df_forecast))

        train_df_forecast = train_df_forecast.reset_index()
        # train_df_forecast.rename(columns={'index':'REFERRAL_MONTH'},inplace=True)
        test_df_forecast = test_df_forecast.reset_index()
        # test_df_forecast.rename(columns={'index':'REFERRAL_MONTH'},inplace=True)

        train_df_forecast['SET'] = 'train'
        test_df_forecast['SET'] = 'test'

        hw_tt_pred_df = pd.DataFrame(hw_tt_pred,columns=['REFERRALS_RECEIVED'])
        hw_tt_pred_df.reset_index(inplace=True)
        hw_tt_pred_df.rename(columns={'index':'REFERRAL_MONTH'},inplace=True)
        hw_tt_pred_df['SET'] = 'predictions'

        df_forecast_tt_plot = pd.concat([train_df_forecast,test_df_forecast,hw_tt_pred_df])

        # Create the Altair plot
        tt_pred_plot = (
                alt.Chart(df_forecast_tt_plot,
                title=f'Train-Test-Predictions for Holt-Winters model with {'no' if trend is None else trend} trend and {'no' if seasonality is None else seasonality} seasonality',
                width=WIDTH, height=HEIGHT)
                .mark_line(point=False)
                .encode(
                        x=alt.X('REFERRAL_MONTH:T',axis=alt.Axis(format='%b-%Y',labelAngle=-90,tickCount='month')),
                        y='REFERRALS_RECEIVED:Q',
                        tooltip=['REFERRALS_RECEIVED'],
                        color=alt.Color('SET:N').scale(scheme={'expr': 'Colour Scheme'})
            ).interactive()
            .add_params(
                SCHEME_DROPDOWN
            )
        )
        
        # Do the Root Mean Squared Error calculation for the model's predictions versus the test set
        rsme = math.sqrt(mean_squared_error(test_df_forecast['REFERRALS_RECEIVED'],hw_tt_pred_df['REFERRALS_RECEIVED']))

        # Get a normalised RSME in order to get a sense of how well the models are performing between
        # modelling service datasets. For this we will use the standard deviation of the dataset as
        # the normalisation method.

        nrmse = rsme / np.std(test_df_forecast['REFERRALS_RECEIVED'])

        # Render the NRSME value

        st.write(f'The Normalised Root Mean Squared Error value for this model is: {nrmse:.2f}') # NRSME rounded to 2 decimal places.

        # Render the Altair plot
        st.altair_chart(tt_pred_plot, use_container_width=False)

    with tab3:
        st.header('Making a prediction with our model')

        st.markdown('''
                    We can now make a prediction using our model. Lorem ipsum...
        '''
        )

        # Apply the model to the whole historic dataset
        hw_prod = ExponentialSmoothing(
            df_forecast['REFERRALS_RECEIVED'],
            trend=trend,
            seasonal=seasonality,
            seasonal_periods=12 # monthly data; year represents a "season" in terms of the cycle of repetition
        )

        forecast = hw_prod.fit().forecast(12)

        forecast = forecast.reset_index().rename(columns={'index':'REFERRAL_MONTH',0:'REFERRALS_RECEIVED'})

        forecast['SET'] = 'forecast'

        df_forecast = df_forecast.reset_index()

        # Create a dataframe of the original historic data, labelled as 'actuals'
        actuals = df_forecast.copy().drop(columns=['PROVIDER','MODELLING_SERVICE'])

        actuals['SET'] = 'actuals'

        hw_forecast_df = pd.concat([actuals,forecast])

        # Create the Altair plot
        forecast_plot = (
                alt.Chart(hw_forecast_df,
                title=f'Forecast for Holt-Winters model with {'no' if trend is None else trend} trend and {'no' if seasonality is None else seasonality} seasonality',
                width=WIDTH, height=HEIGHT)
                .mark_line(point=False)
                .encode(
                        x=alt.X('REFERRAL_MONTH:T',axis=alt.Axis(format='%b-%Y',labelAngle=-90,tickCount='month')),
                        y='REFERRALS_RECEIVED:Q',
                        tooltip=['REFERRALS_RECEIVED'],
                        color=alt.Color('SET:N').scale(scheme={'expr': 'Colour Scheme'})
            ).interactive()
            .add_params(
                SCHEME_DROPDOWN
            )
        )

        # Render the Altair plot
        st.altair_chart(forecast_plot, use_container_width=False)

else:
    st.info('Please upload a .csv file of your data to continue')