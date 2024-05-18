#Importing necessary libraries
from tkinter import *
from PIL import ImageTk, Image  
from tkinter import messagebox
import re
import pickle
import pandas as pd
import joblib
import numpy as np
from tensorflow.keras.models import load_model


#loading the saved model
loaded_model=load_model("Project_Saved_Models/Trained_CNN_model.h5")
#loading standardscaler
scaler=pickle.load(open('Project_Extra/scaler_dl.pkl','rb'))


def check(un_entry,password_entry1):
    username=un_entry.get()
    #print("username : ",username)
    password=password_entry1.get()
    #print("password : ",password)

    if(username=="" or password==""):
        messagebox.showwarning("warning","Please Fill Details")  
    elif(username=="admin" and password=="admin"):
        admin()
    else:
        messagebox.showwarning("warning","Invalid Credentials")  


def login():
    LoginPage = Frame(window)
    LoginPage.grid(row=0, column=0, sticky='nsew')
    LoginPage.tkraise()
    window.title('Cognitive Workload Predictor')

    #login page
    de1 = Listbox(LoginPage, bg='#2f7a61', width=115, height=50, highlightthickness=0, borderwidth=0)
    de1.place(x=0, y=0)
    de2 = Listbox(LoginPage, bg= '#62bd9f', width=115, height=50, highlightthickness=0, borderwidth=0)
    de2.place(x=606, y=0)

    de3 = Listbox(LoginPage, bg='#8be0c4', width=100, height=33, highlightthickness=0, borderwidth=0)
    de3.place(x=76, y=66)

    de4 = Listbox(LoginPage, bg='#f8f8f8', width=85, height=33, highlightthickness=0, borderwidth=0)
    de4.place(x=606, y=66)
    #  Username
    un_entry = Entry(de4, fg="#333333", font=("yu gothic ui semibold", 12), highlightthickness=2,
                        )
    un_entry.place(x=134, y=170, width=256, height=34)
    un_entry.config(highlightbackground="black", highlightcolor="black")
    un_label = Label(de4, text='• Username', fg="#89898b", bg='#f8f8f8', font=("yu gothic ui", 11, 'bold'))
    un_label.place(x=130, y=140)
    #  Password 
    password_entry1 = Entry(de4, fg="#333333", font=("yu gothic ui semibold", 12), show='*', highlightthickness=2,
                            )
    password_entry1.place(x=134, y=250, width=256, height=34)
    password_entry1.config(highlightbackground="black", highlightcolor="black")
    password_label = Label(de4, text='• Password', fg="#89898b", bg='#f8f8f8', font=("yu gothic ui", 11, 'bold'))
    password_label.place(x=130, y=220)

    # function for show and hide password
    def password_command():
        if password_entry1.cget('show') == '*':
            password_entry1.config(show='')
        else:
            password_entry1.config(show='*')

    # checkbutton 
    checkButton = Checkbutton(de4, bg='#f8f8f8', command=password_command, text='show password')
    checkButton.place(x=140, y=288)

    # Welcome Label 
    welcome_label = Label(de4, text='Welcome', font=('Arial', 20, 'bold'), bg='#f8f8f8')
    welcome_label.place(x=130, y=15)

    #top Login Button
    lob = Label(LoginPage, text='Login', font=("yu gothic ui bold", 12), bg='#f8f8f8', fg="#89898b",
                          borderwidth=0, activebackground='#1b87d2')
    lob.place(x=845, y=175)

    lol = Canvas(LoginPage, width=60, height=5, bg='black')
    lol.place(x=836, y=203)

    #  LOGIN  down button 
    loginBtn1 = Button(de4, fg='#f8f8f8', text='Login', bg='#1b87d2', font=("yu gothic ui bold", 15),
                       cursor='hand2', activebackground='#1b87d2',command=lambda:check(un_entry,password_entry1))
    loginBtn1.place(x=133, y=340, width=256, height=50)
    #User icon 
    u_icon = Image.open('images\\user.png')
    photo = ImageTk.PhotoImage(u_icon)
    Uicon_label = Label(de4, image=photo, bg='#f8f8f8')
    Uicon_label.image = photo
    Uicon_label.place(x=103, y=173)

    #  password icon 
    password_icon = Image.open('images\\key.png')
    photo = ImageTk.PhotoImage(password_icon)
    password_icon_label = Label(de4, image=photo, bg='#f8f8f8')
    password_icon_label.image = photo
    password_icon_label.place(x=103, y=253)
    #  picture icon 
    picture_icon = Image.open('images\\user-experience.png')
    photo = ImageTk.PhotoImage(picture_icon)
    picture_icon_label = Label(de4, image=photo, bg='#f8f8f8')
    picture_icon_label.image = photo
    picture_icon_label.place(x=280, y=6)

    #  Left Side Picture 
    side_image = Image.open('images\\home1.jpg')
    side_image = side_image.resize((400,400))
    photo = ImageTk.PhotoImage(side_image)
    side_image_label = Label(de3, image=photo, bg='#ffdb99')
    side_image_label.image = photo
    side_image_label.place(x=70, y=65)


def predict(e_var0,e_var1,e_var2,e_var3,e_var4,e_var5,e_var6,e_var7,e_var8,e_var9):
    
    Wavelet_Approximate_Entropy=e_var0.get()
    Variance_of_Vertex_to_Vertex_Slope=e_var1.get()
    Wavelet_Detailed_Energy=e_var2.get()
    Hjorth_Activity=e_var3.get()
    FFT_Beta_MaxPower=e_var4.get()
    First_Difference_Max=e_var5.get()
    FFT_Alpha_MaxPower=e_var6.get()
    Wavelet_Approximate_Std_Deviation=e_var7.get()
    FFT_Theta_MaxPower=e_var8.get()
    Coeffiecient_of_Variation=e_var9.get()


    if Wavelet_Approximate_Entropy=='' or Variance_of_Vertex_to_Vertex_Slope=='' or Wavelet_Detailed_Energy=='' or Hjorth_Activity=='' or FFT_Beta_MaxPower=='' or First_Difference_Max=='' or FFT_Alpha_MaxPower=='' or Wavelet_Approximate_Std_Deviation=='' or FFT_Theta_MaxPower=='' or Coeffiecient_of_Variation=='':
        messagebox.showwarning("Warning","Please Fill all Fields")
    else:
        info=[]
        parameters=['Wavelet Approximate Entropy','Variance of Vertex to Vertex Slope','Wavelet Detailed Energy',
            'Hjorth_Activity','FFT Beta MaxPower','1st Difference Max','FFT Alpha MaxPower',
            'Wavelet Approximate Std Deviation','FFT Theta MaxPower','Coeffiecient of Variation']


        info.append(Wavelet_Approximate_Entropy)
        info.append(Variance_of_Vertex_to_Vertex_Slope)
        info.append(Wavelet_Detailed_Energy)
        info.append(Hjorth_Activity)
        info.append(FFT_Beta_MaxPower)
        info.append(First_Difference_Max)
        info.append(FFT_Alpha_MaxPower)
        info.append(Wavelet_Approximate_Std_Deviation)
        info.append(FFT_Theta_MaxPower)
        info.append(Coeffiecient_of_Variation)
 
        
        my_dict=dict(zip(parameters,info))

        #convert dict into dataframe
        my_data=pd.DataFrame(my_dict,index=[0])

        feat=np.array(my_data)
        # print(feat)
        #perform standardization
        feat=scaler.transform(feat)
        # print(feat)
        #expand dimension
        feat = np.expand_dims(feat, axis=2)
        # print(feat)
        # print(feat.shape)

        # dataframe is putting into the MODEL to make PREDICTION
        my_pred = loaded_model.predict(feat)
        print(my_pred)
        my_pred=np.argmax(my_pred)
        print(my_pred)

        print("\n*************Result**************")

        if my_pred==0:
            print("Mental Cognitive Workload : Low ")
            output="Mental Cognitive Workload : Low\n\nIn a low cognitive workload scenario, the task at hand is relatively easy or requires minimal mental effort to complete. The individual may experience low levels of stress or cognitive strain."

        if my_pred==1:
            print("Mental Cognitive Workload : Medium ")
            output="Mental Cognitive Workload : Medium\n\nA medium cognitive workload signifies that the task requires a moderate level of mental effort. The individual may need to focus and concentrate to complete the task efficiently."

        if my_pred==2:
            print("Mental Cognitive Workload : High ")
            output="Mental Cognitive Workload : High\n\nHigh cognitive workload indicates that the task demands significant mental resources and concentration. The individual may experience heightened stress or mental fatigue as they work to manage complex information or tasks."


        messagebox.showinfo("Result",output)

        # out_label.config(text=output)



    
def admin():
    Admin=Frame(window,bg="#5845d3")
    Admin.grid(row=0, column=0, sticky='nsew')
    Admin.tkraise()
    window.title('Cognitive Workload Predictor')


    de2 = Listbox(Admin, bg='#ebdeae', width=200, height=42, highlightthickness=0, borderwidth=0)
    de2.place(x=0, y=0)

    input_label = Label(de2, text='Input', font=('Arial', 24, 'bold'), bg='#ebdeae')
    input_label.place(x=545, y=38)
    i1 = Canvas(de2, width=104, height=2, bg='#333333',highlightthickness=0)
    i1.place(x=530, y=82)


    label1 = Label(de2, text="Wavelet Approximate Entropy :",
                   font="arial 12 bold", bg="#ebdeae")
    label1.place(x=80, y=150)
    label2 = Label(de2, text="Variance of V to V Slope :",
                   font="arial 12 bold", bg="#ebdeae")
    label2.place(x=80, y=200)
    label3 = Label(de2, text="Wavelet Detailed Energy :",
                   font="arial 12 bold", bg="#ebdeae")
    label3.place(x=80, y=250)
    label4 = Label(de2, text="Hjorth_Activity :",
                   font="arial 12 bold", bg="#ebdeae")
    label4.place(x=80, y=300)
    label5 = Label(de2, text="FFT Beta MaxPower :",
                   font="arial 12 bold", bg="#ebdeae")
    label5.place(x=80, y=350)
    label6 = Label(de2, text="1st Difference Max :",
                   font="arial 12 bold", bg="#ebdeae")
    label6.place(x=660, y=150)
    label7 = Label(de2, text="FFT Alpha MaxPower :",
                   font="arial 12 bold", bg="#ebdeae")
    label7.place(x=660, y=200)
    label8 = Label(de2, text="Wavelet Approximate StdD :",
                   font="arial 12 bold", bg="#ebdeae")
    label8.place(x=660, y=250)
    label9 = Label(de2, text="FFT Theta MaxPower :",
                   font="arial 12 bold", bg="#ebdeae")
    label9.place(x=660, y=300)
    label10 = Label(de2, text="Coeffiecient of Variation :", font="arial 12 bold", bg="#ebdeae")
    label10.place(x=660, y=350)
    

    global e_var0,e_var1,e_var2,e_var3,e_var4,e_var5,e_var6,e_var7,e_var8,e_var9
    e_var0=StringVar()
    e_var1=StringVar()
    e_var2=StringVar()
    e_var3=StringVar()
    e_var4=StringVar()
    e_var5=StringVar()
    e_var6=StringVar()
    e_var7=StringVar()
    e_var8=StringVar()
    e_var9=StringVar()

    entry0 = Entry(de2, textvariable=e_var0, bd=2, width=25)
    entry0.place(x=330, y=150)
    entry1 = Entry(de2, textvariable=e_var1, bd=2, width=25)
    entry1.place(x=330, y=200)
    entry2 = Entry(de2, textvariable=e_var2, bd=2, width=25)
    entry2.place(x=330, y=250)
    entry3 = Entry(de2, textvariable=e_var3, bd=2, width=25)
    entry3.place(x=330, y=300)
    entry4 = Entry(de2, textvariable=e_var4, bd=2, width=25)
    entry4.place(x=330, y=350)

    entry5 = Entry(de2, textvariable=e_var5, bd=2, width=25)
    entry5.place(x=890, y=150)
    entry6 = Entry(de2, textvariable=e_var6, bd=2, width=25)
    entry6.place(x=890, y=200)
    entry7 = Entry(de2, textvariable=e_var7, bd=2, width=25)
    entry7.place(x=890, y=250)
    entry8 = Entry(de2, textvariable=e_var8, bd=2, width=25)
    entry8.place(x=890, y=300)
    entry9 = Entry(de2, textvariable=e_var9, bd=2, width=25)
    entry9.place(x=890, y=350)


  
    # Buttons
    p_b_image=Image.open('images\\pre_button.png')
    p_b_photo=ImageTk.PhotoImage(p_b_image)
    predict_Btn1 = Button(de2, image=p_b_photo, bg='#ebdeae',
                       cursor='hand2',bd=0, activebackground='#6699ff',command=lambda:predict(e_var0,e_var1,e_var2,e_var3,e_var4,e_var5,e_var6,e_var7,e_var8,e_var9))
    predict_Btn1.image=p_b_photo
    predict_Btn1.place(x=383, y=530)


    refresh_image=Image.open('images\\re_button.png')
    refresh_photo=ImageTk.PhotoImage(refresh_image)
    refresh_Btn1 = Button(de2, image=refresh_photo, bg='#ebdeae',
                       cursor='hand2',bd=0, activebackground='#00cc66',command=lambda:admin())
    refresh_Btn1.image=refresh_photo
    refresh_Btn1.place(x=653, y=530)

    
window = Tk()
window.rowconfigure(0, weight=1)
window.columnconfigure(0, weight=1)
window.geometry("1200x650")
window.maxsize(1200, 650)
window.minsize(1200, 650)
# Window Icon Photo
icon = PhotoImage(file='images\\pic-icon.png')
window.iconphoto(True, icon)
login()
# admin()

window.mainloop()
