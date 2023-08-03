# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 02:09:49 2023

@author: amaru
"""

import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


from imblearn.over_sampling import SMOTE

research_data = pd.read_excel('D:/Diss/bicycleUpdated.xlsx')
research_data.info()

research_data = research_data.set_index('dates').sort_index()

#p-value using chi-squared test
vehicle_severity = pd.crosstab(research_data['number_of_vehicles'], research_data['severity'])
casualties_severity = pd.crosstab(research_data['number_of_casualties'], research_data['severity'])
speed_severity = pd.crosstab(research_data['speed_limit'], research_data['severity'])
road_conditions_severity = pd.crosstab(research_data['road_conditions'], research_data['severity'])
weather_conditions_severity = pd.crosstab(research_data['weather_conditions'], research_data['severity'])
road_type_severity = pd.crosstab(research_data['road_type'], research_data['severity'])
light_conditions_severity = pd.crosstab(research_data['road_conditions'], research_data['severity'])

chi2, p, dof, expected = chi2_contingency(vehicle_severity)
chi2, p, dof, expected = chi2_contingency(casualties_severity)
chi2, p, dof, expected = chi2_contingency(road_conditions_severity)
chi2, p, dof, expected = chi2_contingency(weather_conditions_severity)
chi2, p, dof, expected = chi2_contingency(road_type_severity)
chi2, p, dof, expected = chi2_contingency(light_conditions_severity)
print("P-Value: ", p)


research_data_encoded = research_data.apply(LabelEncoder().fit_transform)

depend_var = research_data_encoded['severity']
independent_vars = research_data_encoded.drop('severity', axis = 1)

independent_vars_train, independent_vars_test, depend_var_train, depend_var_test = train_test_split(independent_vars, depend_var, test_size=0.10, random_state=1)

smote = SMOTE(random_state=42)

independent_vars_train_resampled, depend_var_train_resampled = smote.fit_resample(independent_vars, depend_var)


#NaiveBayes
nb_model = GaussianNB()
nb_model.fit(independent_vars_train_resampled, depend_var_train_resampled)
nb_model_pred = nb_model.predict(independent_vars_test)

print(classification_report(depend_var_test, nb_model_pred))
confusion_matrix(depend_var_test, nb_model_pred)
accuracy_score(depend_var_test, nb_model_pred)#73.33%


#RandomForest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(independent_vars_train_resampled, depend_var_train_resampled)
rf_model_pred = rf_model.predict(independent_vars_test)

print(classification_report(depend_var_test, rf_model_pred))
confusion_matrix(depend_var_test, rf_model_pred)
accuracy_score(depend_var_test, rf_model_pred)#72.84%


#DecisionTree
decisionTreeModel = DecisionTreeClassifier(criterion = 'entropy', 
                                           random_state = 100, 
                                           max_depth = 7, 
                                           min_samples_leaf = 8)
decisionTreeModel.fit(independent_vars_train_resampled, depend_var_train_resampled)
pred_dt = decisionTreeModel.predict(independent_vars_test)

print(classification_report(depend_var_test, pred_dt))
confusion_matrix(depend_var_test, pred_dt)
accuracy_score(depend_var_test, pred_dt)#69.58



# Data Exploration
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

accidents = pd.read_csv('D:/Diss/Accidents.csv')
accidents.info()

accidents['dates'] = pd.to_datetime(accidents['Date'])

accidents['years'] = accidents['dates'].dt.year

ac_filtered = accidents[accidents['years'] > 2002]
ac_filtered = ac_filtered.sort_index()

ocurrence_counts = ac_filtered['years'].value_counts()
ocurrence_counts = ocurrence_counts.sort_index()

ocurrence_counts = ocurrence_counts.to_frame()

ocurrence_counts['casualties'] = ac_filtered.groupby(['years'])['Number_of_Casualties'].sum()

ocurrence_counts = ocurrence_counts.rename(columns={'number of crashes': 'Number of crashes'})
ocurrence_counts = ocurrence_counts.rename(columns={'casualties': 'Number of casualties'})

year_plot = ocurrence_counts.plot(kind='bar', width=0.8)
year_plot.set_xlabel('Year')
year_plot.set_ylabel('Number of crashes and casualties')
year_plot.set_title('Number of Crashes and Casualties')
plt.show()

ac_filtered['months'] = ac_filtered['dates'].dt.month

monthly_data = ac_filtered['months'].value_counts()
monthly_data = monthly_data.sort_index()

monthly_data = monthly_data.to_frame()

monthly_data = monthly_data.rename(columns={'months': 'Monthly crashes'})

monthly_plot = monthly_data.plot(kind='line')
monthly_plot.set_xlabel('Month')
monthly_plot.set_ylabel('Number of crashes')
monthly_plot.set_title('Number of Crashes Each Month')
monthly_plot.get_legend().remove()
plt.show()

zero_and_one = ac_filtered[(ac_filtered['Time'] >= '00:00') & (ac_filtered['Time'] < '01.00')]
one_and_two = ac_filtered[(ac_filtered['Time'] >= '01:00') & (ac_filtered['Time'] < '02.00')]
two_and_three = ac_filtered[(ac_filtered['Time'] >= '02:00') & (ac_filtered['Time'] < '03.00')]
three_and_four = ac_filtered[(ac_filtered['Time'] >= '03:00') & (ac_filtered['Time'] < '04.00')]
four_and_five = ac_filtered[(ac_filtered['Time'] >= '04:00') & (ac_filtered['Time'] < '05.00')]
five_and_six = ac_filtered[(ac_filtered['Time'] >= '05:00') & (ac_filtered['Time'] < '06.00')]
six_and_seven = ac_filtered[(ac_filtered['Time'] >= '06:00') & (ac_filtered['Time'] < '07.00')]
seven_and_eight = ac_filtered[(ac_filtered['Time'] >= '07:00') & (ac_filtered['Time'] < '08.00')]
eight_and_nine = ac_filtered[(ac_filtered['Time'] >= '08:00') & (ac_filtered['Time'] < '09.00')]
nine_and_ten = ac_filtered[(ac_filtered['Time'] >= '09:00') & (ac_filtered['Time'] < '10.00')]
ten_and_eleven = ac_filtered[(ac_filtered['Time'] >= '10:00') & (ac_filtered['Time'] < '11.00')]
eleven_and_twelve = ac_filtered[(ac_filtered['Time'] >= '11:00') & (ac_filtered['Time'] < '12.00')]
twelve_and_thirteen = ac_filtered[(ac_filtered['Time'] >= '12:00') & (ac_filtered['Time'] < '13.00')]
thirteen_and_fourteen = ac_filtered[(ac_filtered['Time'] >= '13:00') & (ac_filtered['Time'] < '14.00')]
fourteen_and_fifteen = ac_filtered[(ac_filtered['Time'] >= '14:00') & (ac_filtered['Time'] < '15.00')]
fifteen_and_sixteen = ac_filtered[(ac_filtered['Time'] >= '15:00') & (ac_filtered['Time'] < '16.00')]
sixteen_and_seventeen = ac_filtered[(ac_filtered['Time'] >= '16:00') & (ac_filtered['Time'] < '17.00')]
seventeen_and_eighteen = ac_filtered[(ac_filtered['Time'] >= '17:00') & (ac_filtered['Time'] < '18.00')]
eighteen_and_nineteen = ac_filtered[(ac_filtered['Time'] >= '18:00') & (ac_filtered['Time'] < '19.00')]
nineteen_and_twenty = ac_filtered[(ac_filtered['Time'] >= '19:00') & (ac_filtered['Time'] < '20.00')]
twenty_and_twentyOne = ac_filtered[(ac_filtered['Time'] >= '20:00') & (ac_filtered['Time'] < '21.00')]
twentyOne_and_twentyTwo = ac_filtered[(ac_filtered['Time'] >= '21:00') & (ac_filtered['Time'] < '22.00')]
twentyTwo_and_twentyThree = ac_filtered[(ac_filtered['Time'] >= '22:00') & (ac_filtered['Time'] < '23.00')]
twentyThree_and_zero = ac_filtered[(ac_filtered['Time'] >= '23:00') & (ac_filtered['Time'] <= '23:59')]

data = {'Hours': ['zero_and_one', 'one_and_two', 'two_and_three', 'three_and_four', 'four_and_five', 
                 'five_and_six', 'six_and_seven', 'seven_and_eight', 'eight_and_nine', 'nine_and_ten', 'ten_and_eleven',
                 'eleven_and_twelve', 'twelve_and_thirteen', 'thirteen_and_fourteen', 'fourteen_and_fifteen',
                 'fifteen_and_sixteen', 'sixteen_and_seventeen', 'seventeen_and_eighteen', 'eighteen_and_nineteen',
                 'nineteen_and_twenty', 'twenty_and_twentyOne', 'twentyOne_and_twentyTwo', 'twentyTwo_and_twentyThree',
                 'twentyThree_and_zero'],
        'Number of crashes': [1630, 836, 566, 434, 462, 2170, 6721, 17708, 26887, 14304, 10639, 11151, 12513, 13072, 13297, 17969, 22026, 28351, 22929, 15442, 9325, 6005, 4487, 2889]}


#hourly_data = data.rename(columns={'zero_and_one':'00:00-00:59', 'one_and_two':'01:00-01:59', 'two_and_three':'02:00-02:59', 'three_and_four':'03:00-03:59', 'four_and_five':'04:00-04:59', 
 #                'five_and_six':'05:00-05:59', 'six_and_seven':'06:00-06:59', 'seven_and_eight':'07:00-07:59', 'eight_and_nine':'08:00-08:59', 'nine_and_ten':'09:00-09:59', 'ten_and_eleven':'10:00-10:59',
  #               'eleven_and_twelve':'11:00-11:59', 'twelve_and_thirteen':'12:00-12:59', 'thirteen_and_fourteen':'13:00-13:59', 'fourteen_and_fifteen':'14:00-14:59',
   #              'fifteen_and_sixteen':'15:00-15:59', 'sixteen_and_seventeen':'16:00-16:59', 'seventeen_and_eighteen':'17:00-17:59', 'eighteen_and_nineteen':'18:00-18:59',
    #             'nineteen_and_twenty':'19:00-19:59', 'twenty_and_twentyOne':'20:00-20:59', 'twentyOne_and_twentyTwo':'21:00-21:59', 'twentyTwo_and_twentyThree':'22:00-22:59',
     #            'twentyThree_and_zero':'23:00-23:59'})


hourly_data = pd.DataFrame(data)


bikers = pd.read_csv('D:/Diss/bikers_new.csv')

merged_df = pd.merge(ac_filtered, bikers, on='Accident_Index')

age_data = merged_df['Age_Grp'].value_counts()
age_data = pd.DataFrame(age_data)
age_data['Age Group'] = age_data.index

age_plot = age_data.plot(kind='bar')
age_plot.set_xlabel('Age Group')
age_plot.set_ylabel('Number of crashes')
age_plot.set_title('Number of Crashes by Age-Group')
age_plot.get_legend().remove()
plt.show()


gender_data = merged_df['Gender'].value_counts()
gender_data = pd.DataFrame(gender_data)
gender_data = gender_data.rename(columns={'Gender': 'Crashes by gender'})
gender_data['Gender'] = gender_data.index

gender_plot = gender_data.plot(kind='bar')
gender_plot.set_xlabel('Gender')
gender_plot.set_ylabel('Number of crashes')
gender_plot.set_title('Number of Crashes by Gender')
gender_plot.get_legend().remove()
plt.show()

roadType_data = merged_df['Road_type'].value_counts()
roadType_data = pd.DataFrame(roadType_data)
roadType_data = roadType_data.rename(columns={'Road_type': 'Crashes by Road Type'})
roadType_data['Road Type'] = roadType_data.index

roadType_plot = roadType_data.plot(kind='bar')
roadType_plot.set_xlabel('Road Type')
roadType_plot.set_ylabel('Number of crashes')
roadType_plot.set_title('Number of Crashes by Road Type')
roadType_plot.get_legend().remove()
plt.show()


newdf = merged_df.drop(merged_df[merged_df['Road_conditions'] == 'Missing Data'].index)

Weekly_data = merged_df['Day'].value_counts()
Weekly_data = pd.DataFrame(Weekly_data)
Weekly_data = Weekly_data.rename(columns={'Day': 'Crashes by Road Condition'})
Weekly_data['Day'] = Weekly_data.index

Weekly_plot = Weekly_data.plot(kind='line')
Weekly_plot.set_xlabel('Day')
Weekly_plot.set_ylabel('Number of crashes')
Weekly_plot.set_title('Number of Crashes by Day of Week')
Weekly_plot.get_legend().remove()
plt.show()



ml_accuracy = {'ML Model': ['NB', 'RF', 'DT', 'FFNN', 'KNN-XGBoost'], 
               'Accuracy': ['73.33', '72.84', '69.58', '69.24', '83.56']}

ml_accuracy = pd.DataFrame(ml_accuracy)
ml_accuracy['Accuracy'] = ml_accuracy['Accuracy'].astype(float)

ml_accuracy_plot = ml_accuracy.plot(kind='bar')
ml_accuracy_plot.set_xlabel('Model')
ml_accuracy_plot.set_ylabel('Accuracy')
ml_accuracy_plot.set_title('Accuracy rate of each model')
ml_accuracy_plot.set_xticks(range(len(ml_accuracy['ML Model'])))
ml_accuracy_plot.set_xticklabels(ml_accuracy['ML Model'])
plt.show()








