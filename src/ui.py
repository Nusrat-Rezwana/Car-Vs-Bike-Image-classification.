from customtkinter import *
from PIL import Image

from predict import *

app = CTk()
app.geometry("900x700")

image_file = '../data/test/Bike/Bike (1070).jpeg'

def selectfile():
    filename=filedialog.askopenfilename()
    print(filename)
    global image_file
    image_file=filename
    img=Image.open(filename)
    image=CTkImage(light_image=img,dark_image=img,size=(300,400))
    imLabel=CTkLabel(app,text="",image=image)
    imLabel.place(relx = 0.5, rely = 0.5,anchor="center")



def classify():
    model = CarBikeClassifier(num_classes=2)
    model.load_state_dict(torch.load('../pretrained_models/model.pth', map_location=torch.device('cpu')))
    print(image_file)
    prediction_text=predict_image(model, image_file, device='cpu')

    frame=CTkFrame(master=app,fg_color="transparent",border_color="white",border_width=2)
    frame.place(relx = 0.5, rely = 0.1,anchor="center")
    txt=CTkLabel(master=frame,text=prediction_text,font=("Roboto",40),pady=5,padx=5)
    txt.pack(anchor="s",expand=True,pady=3,padx=3)



button_to_select = CTkButton(master=app, text = "Select image", fg_color = "black", command = selectfile)
button_to_select.pack(padx = 5, pady = 5)
button_to_select.place(relx = 0.4, rely = 0.9,anchor="center")

classify_button=CTkButton(master=app,text="Classify",fg_color="black",command=classify)
classify_button.place(relx=0.6,rely=0.9,anchor="center")

app.mainloop()


