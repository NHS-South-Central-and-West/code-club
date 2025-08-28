import pandas as pd
import streamlit as st
import altair as alt
import datetime as dt
alt.renderers.enable('default')
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import math
import numpy as np
from sklearn.metrics import mean_squared_error
from prophet import Prophet
from prophet.plot import (plot_plotly, 
                          plot_components_plotly,
                          plot_forecast_component,
                          add_changepoints_to_plot)
import matplotlib.pyplot as plt

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

    tab1,tab2,tab3,tab4 = st.tabs(['Background Information','Holt-Winters Train-Test','Holt-Winters Predictions','Prophet Forecast'])

    with tab1:

        st.markdown('''
                    This demonstration is intended to show how it is possible to have
                    some form of modelling technique set up that can work with data
                    uploaded by the end user. **It should not be viewed as a demonstration
                    of a full forecasting workflow.**

                    First of all, it will demonstrate the use of a Holt-Winters forecasting 
                    model from the `statsmodels` [package](https://www.statsmodels.org/stable/index.html).
                    This is an example of a forecasting model built on a traditional statistical
                    technique. It is based on an Exponential Smoothing model, which creates forecasts
                    by taking a weighted average of past observations, giving more weight (i.e. importance)
                    to more recent observations. "Exponential" comes from the fact that the weight
                    descreases exponentially as the observations get older. However, the Exponential
                    Smoothing model is suited to data that show no trend or seasonality. The Holt-Winters
                    model adds trend and seasonality elements to the model, so it can be applied to time
                    series that show trend and seasonality. This demonstration will allow the users to set 
                    the trend and seasonality parameters.

                    In addition to this, there is a demonstration of Facebook's [Prophet](https://facebook.github.io/prophet/) 
                    forecasting model, which is an example of a machine learning approach to forecasting. 
                    It is a very effective method for getting a competent forecast "out of the box", without 
                    the need to think about any parameters.
        '''
        )

    with tab2:

        st.header('Training and Testing the Holt-Winters Model')

        st.markdown('''
                    Here we can select the type of trend component and kind of seasonality
                    component for our Holt-Winters forecasting model, and then see which 
                    combination performs best at making a prediction against our test set.

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
                options= mod_serv_options,
                key=111
            )

        with ttcol2:
            trend = st.selectbox(
            label='Trend',
            options=[None,'additive','multiplicative'],
            key=112
        )
        with ttcol3:
            seasonality = st.selectbox(
                label='Seasonality',
                options=[None,'additive','multiplicative'],
                key=113
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
        st.header('Making a prediction with our Holt-Winters model')

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
    
    with tab4:
        st.header('Using the Prophet model to make a forecast')

        mod_serv_options2 = {val for val in df_mod['MODELLING_SERVICE'] if len(df_mod[df_mod['MODELLING_SERVICE'] == val]) > 29}
        mod_serv_select2 = st.selectbox(
            label = 'Modelling Service',
            options= mod_serv_options2,
            key=114
        )

        # Use this utility function to help format the data for Prophet
        def _prophet_training_data(y_train):
            '''
            Courtesy of Dr. Tom Monks, University of Exeter
            ---------
            Converts a standard pandas datetimeindexed dataframe
            for time series into one suitable for Prophet
            Parameters:
            ---------
            y_train: pd.DataFrame
                univariate time series data
                
            Returns:
            --------
                pd.DataFrame in Prophet format 
                columns = ['ds', 'y']
            '''
            prophet_train = pd.DataFrame(y_train.index)
            prophet_train['y'] = y_train.to_numpy()
            prophet_train.columns = ['ds', 'y']

            return prophet_train
        

        df_forecast2 = df_mod[df_mod['MODELLING_SERVICE'] == mod_serv_select2].copy()

        df_forecast2.set_index('REFERRAL_MONTH',inplace=True)

        # Create train and test datasets
        train_df_forecast2 = df_forecast2['REFERRALS_RECEIVED'].iloc[:TRAIN_SIZE]
        test_df_forecast2 = df_forecast2['REFERRALS_RECEIVED'].iloc[TRAIN_SIZE:]

        # Index the data as month start
        train_df_forecast2.index.freq = 'MS'

        # Format the training data for Prophet with the utility function

        prophet_train = _prophet_training_data(train_df_forecast2)

        # Fit model
        p_model = Prophet(interval_width = 0.95, seasonality_mode ='additive', seasonality_prior_scale=2.0)
        p_model.fit(prophet_train)

        # Make a prediction and analyse components

        future = p_model.make_future_dataframe(periods = 12, freq = 'MS')
        prophet_forecast = p_model.predict(future)

        fig, ax = plt.subplots(figsize=(6, 4))
        p_model.plot(prophet_forecast, xlabel='Date', ylabel='Value', ax=plt.gca())
        test_df_forecast2.plot(color = 'red', label = 'Test data (actuals)')
        plt.title(f'Prophet Forecast - {mod_serv_select2} Modelling Service')
        plt.legend()

        st.pyplot(fig, use_container_width=False)

else:
    st.info('Please upload a .csv file of your data to continue')