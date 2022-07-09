from sklearn.model_selection import train_test_split
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
matplotlib.use('Agg')
from PIL import Image

#Set title

st.title("Simple ML App using Streamlit")
image=Image.open('abc.jpeg')
st.image(image,use_column_width=True)


def main():
    activities=['EDA','Visualisation','Model','About us']
    option=st.sidebar.selectbox('Selection option : ', activities)
    
    #Dealing with Data Analysis
    if option=='EDA':
        st.subheader("Exploratory Data Analysis")
        data=st.file_uploader("Upload dataset: ",type=['csv','xlsx','txt','json'])
        st.success("Data successfully loaded")
        if data is not None:
            df=pd.read_csv(data)
            st.dataframe(df.head(20))
            if st.checkbox("Display Shape of Dataset"):
                st.write(df.shape)
            if st.checkbox("Display Columns"):
                st.write(df.columns)
            if st.checkbox("Select multiple columns"):
                selected_columns=st.multiselect('Select preferred columns: ',df.columns)
                df1=df[selected_columns]
                st.dataframe(df1)
            if st.checkbox("Display Summary"):
                st.write(df.describe().T)
            if st.checkbox('Display Null Values'):
                st.write(df.isnull().sum())
            if st.checkbox("Display the data types"):
                st.write(df.dtypes)
            if st.checkbox("Display Correlation of data various columns"):
                st.write(df.corr)

    #Dealing with the VISUALIZATION   
    elif option=='Visualisation':
        st.subheader("Data Visualisation")

        data=st.file_uploader("Upload dataset: ",type=['csv','xlsx','txt','json'])
        st.success("Data successfully loaded")
        if data is not None:
            df=pd.read_csv(data)
            st.dataframe(df.head(20))

            if st.checkbox('Select Multiple Columns to Plot'):
                selected_columns=st.multiselect('Select your preferred columns',df.columns)
                df1=df[selected_columns]
                st.dataframe(df1)
            if st.checkbox('Display Heatmap'):
                fig = plt.figure(figsize=(10, 4))
                st.write(sns.heatmap(df.corr(),vmax=1,square=True,annot=True,cmap='viridis'))
                st.pyplot(fig)
            #if st.checkbox('Display Pairplot'):
                #fig=plt.figure(figsize=(10,4))
                #st.write(sns.pairplot(df,diag_kind='kde'))
                #st.pyplot(fig)
            #if st.checkbox('Display Pie Chart'):
                #all_columns=df.columns.to_list()
                #pie_columns=st.selectbox("Select column to display",all_columns)
                #pieChart=df[pie_columns].value_counts().plot.pie(autopct="%1.1f%%")
                #fig=plt.figure(figsize=(10,4))
                #st.write(pieChart)
                #st.pyplot(fig)

 #DEALING WITH THE MODEL BUILDING
    elif option=='Model':
        st.subheader("Model Building")

        data=st.file_uploader("Upload dataset: ",type=['csv','xlsx','txt','json'])
        st.success("Data successfully loaded")
        if data is not None:
            df=pd.read_csv(data)
            st.dataframe(df.head(10))
            

            #if st.checkbox('Select Multiple Columns'):
                #new_data=st.multiselect("Select your preferred columns. NB:Let your target variable be the last column to be selected",df.columns)
                #df1=df[new_data]
                #st.dataframe(df1)
                #Dividing my data into X and y variables
            X=df.iloc[:,0:-1]
            y=df.iloc[:,-1]
            seed = st.sidebar.slider('Seed',1,200)
            classifier_name=st.sidebar.selectbox('Select your preferred classifier: ',('SVM','KNN','LR','Naive_Bayes','Decision Tree','Random Forest'))
            def add_parameter(name_of_clf):
                params=dict()
                if name_of_clf=='SVM':
                    C=st.sidebar.slider('C',0.01,14.0)
                    params['C']=C
                if name_of_clf=='KNN':
                    K=st.sidebar.slider('K',1,15)
                    params['K']=K
                return params
            #calling the function
            params=add_parameter(classifier_name)


            #defining a function for our classifier
            def get_classifier(name_of_clf,params):
                clf=None
                if name_of_clf=='SVM':
                    clf=SVC(C=params['C'])
                elif name_of_clf=='KNN':
                    clf=KNeighborsClassifier(n_neighbors=params['K'])
                elif name_of_clf=='LR':
                    clf=LogisticRegression()
                elif name_of_clf=='Naive_Bayes':
                    clf=GaussianNB()
                elif name_of_clf=='Decision Tree':
                    clf=DecisionTreeClassifier()
                elif name_of_clf=='Random Forest':
                    clf=RandomForestClassifier()
                else:
                    st.warning('Kindly select preferred algorithm')
                return clf
            clf=get_classifier(classifier_name,params)
            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=seed)
            clf.fit(X_train,y_train)
            y_pred=clf.predict(X_test)
            st.write('Predictions : ',y_pred)

            accuracy=accuracy_score(y_test,y_pred)
            st.write('Name of classifier: ',classifier_name)
            st.write('Accuracy',accuracy)
            
  #Dealing with about us          
    elif option=='About us':
        st.write('#### This is an interactive web page for our ML project , feel free to use it!!!!')
        st.write('##### The datasets used here are fetched from UCI Machine Learning Repository')
        st.write('The analysis in here is to demonstrate how we can present our work to Others in an interesting manner.')
        st.markdown('<br><br>',True)
        st.write("""#### THANK YOU""")
        st.balloons()
if __name__=='__main__':
    main()