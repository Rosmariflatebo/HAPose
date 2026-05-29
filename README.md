
# HAPose
### Instructions for running the program

&nbsp;
## Virtual environment
#### 1. Create your virtual environment
```bash
python -m venv venv
````
#### 2. Activate your virtual environment
##### Windows:
```bash
venv\Scripts\activate
````
##### Mac/Linux:
```bash
source venv/bin/activate
````

&nbsp;
## Programs that need to be installed before running the program:
#### (Make sure that these are installed inside the venv)
#### If any program recommends upgrading any of the software, so as such.
#### 1. Torch
```bash
pip install torch
````
#### 2. Torch Transformers
```bash
pip install torch transformers
````
#### 3. Pandas
```bash
pip install pandas
````
#### 4. Pyserial
```bash
pip install pyserial
````
#### 5. Matplotlib
```bash
pip install matplotlib
````
#### 6. Tkinter
##### Tkinter is already installed in Python on Windows and Mac
##### If Linux is being used install Tkinter with this line:
```bash
sudo apt install python3-tk
```
&nbsp;
## Running the code
#### Run the code from the main.py file
#### Press the "play" button in the top right of the screen (python)
#### Or run the code directly in the terminal:
```bash
python main.py
```

&nbsp;
## Demo vs. actual vest
#### The program runs in a demo format; the reasoning for this is that the values from the code are imported from the actual vest. 
#### This cannot be done without wearing the actual vest.
#### The code instead uses makeshift values that represent the ones that would be extracted from the vest, so that you get a replica of how the application will work when using the actual vest.
#### If you want to run the program with the actual vest:
##### Set the demo_game to false:
###### (This is in line 6 in the main.py file)



