import numpy as np
import pandas as pd
import pickle
import streamlit as st


#copying data from the .pkl file to the app
with open(r"Model_GYM.pkl",'rb') as a_input:
    app =  pickle.load(a_input)
#print(app)
#app =  pickle.load(open("Model_GYM.pkl"),"rb")

#prediction
def predict_output(age,height,weight):
    data = np.array([age,height,weight])
    data = data.reshape(1,-1)
    output_data = app.predict(data)
    return output_data[0]

#print(predict_output(40,5.6,70))


#streamlit code
def main():
    #title
    st.title("Gym App Prediction")
    
    #taking inputs from the webpage
    age = st.text_input("Enter your age: ")
    height = st.text_input("Enter your height in ft ex: 5.4 :")
    weight = st.text_input("Enter your Weight in Kg: ")
    
    requirements = " "
    
    if st.button("Predict"):
        requirements = predict_output(age, height, weight)
        
        if requirements=="Underweight":
            st.info("You're Under Weight")
        elif requirements=="Healthy":
            st.success("You're Healthy")
        elif requirements=="Overweight":
            st.warning("You're Over Weight")
        elif requirements=="Obese":
            st.error("Obesity!")
        else:
            st.error("Extremely Obese")
            
    else:
        st.subheader("Health is Wealth!!")
        st.write("Thank you")
        st.write("Please Visit Again!!")
        

if __name__ ==  "__main__":
    main()
        