import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
st.set_page_config(layout="wide")



header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

# st.markdown(

# 	"""
# 	<style>
# 	.main{

# 	background-color: #F5F5F5;
# 	}
# 	</style>
# 	""",
# 	unsafe_allow_html = True
# 	)

@st.cache_data
def get_data(filename):
	taxi_data = pd.read_csv(filename)

	return taxi_data


with header:
	st.title('Welcome to this Project')
	st.text('In this project, I look into the Solar Prediction dataset')


with dataset:
	st.header('Solar Prediction Dataset')
	st.text('This dataset was taken from https://www.kaggle.com/datasets/dronio/SolarEnergy')

	taxi_data = get_data('data/SolarPrediction.csv')
	st.write(taxi_data.head(20))


	st.subheader('Temperature distribution on Solar Prediction dataset')
	Tempr_dist = taxi_data['Temperature'].value_counts()
	st.bar_chart(Tempr_dist)


with features:
	st.header('Features I created')

	st.markdown('* **First Feature** None ')
	st.markdown('* **Second feature:** None')




with model_training:
	st.header('Time to train models')
	st.text('Here you get to choose the hyperparameters of the model and see how the performance changes')
	sel_col, disp_col = st.columns([1,1])

	max_depth = sel_col.slider('What should be the max_depth of the model?', min_value = 10, max_value = 100, step = 10)
	n_estimators = sel_col.selectbox('How many trees should there be?', options = [100, 200,300,'No Limit'], index = 0)


	sel_col.text('Here are a list of features present in my dataset:')
	sel_col.write(taxi_data.columns)


	input_feature = sel_col.text_input('Which feature should be sued as the input feature?','Temperature')

	if n_estimators == 'No Limit':
		regr = RandomForestRegressor(max_depth = max_depth)
	else:
		regr =  RandomForestRegressor(max_depth = max_depth, n_estimators = n_estimators)

	

	

	X = taxi_data[[input_feature]]
	y = taxi_data[['Humidity']]

	regr.fit(X,y)
	prediction = regr.predict(y)

	disp_col.subheader('Mean absolute error of the model is: ')
	disp_col.write(mean_absolute_error(y, prediction))

	disp_col.subheader('Mean squared error of the model is: ')
	disp_col.write(mean_squared_error(y, prediction))

	disp_col.subheader('R squared error of the model is: ')
	disp_col.write(r2_score(y, prediction))





