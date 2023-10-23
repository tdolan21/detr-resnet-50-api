REM Change to the api-demo-application directory
cd api-demo-application

REM Execute the streamlit command in the background
start streamlit run app.py

REM Change back to the previous directory
cd ..

REM Execute the uvicorn command
uvicorn api:app --reload
