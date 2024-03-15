import pandas as pd

# Define the data
data = {
    "Algorithm": [
        "Greedy", 
        "Logistic Regression", 
        "Random Forest", 
        "Neural Network", 
        "Neural Network w/ Dropout",
        "Neural Network w/ Convolution", 
        "Neural Network (Statistics)", 
        "", 
        "Neural Network w/ Dropout (Statistics)",
        "", 
        "Neural Network w/ Convolution (Statistics)", 
        "", 
        "LSTM", 
        "LSTM (Statistics)", 
        "", 
        "Training"
    ],
    "Accuracy": [
        "57.5%", 
        "72.79%", 
        "72.9%", 
        "73.91%", 
        "73.91%", 
        "73.99%",
        "Max: 74.2%\nMin: 49.5%\nMedian: 64.6%\nAverage: 64.5%", 
        "",
        "Max: 75.0%\nMin: 70.3%\nMedian: 73.0%\nAverage: 72.8%", 
        "",
        "Max: 75.5%\nMin: 71.0%\nMedian: 73.3%\nAverage: 73.1%", 
        "",
        "73.60%", 
        "Max: 76.6%\nMin: 71.1%\nMedian: 74.0%\nAverage: 73.9%", 
        "", 
        "72.75%"
    ],
    "Loss": [
        "N/A", 
        "0.549", 
        "0.542", 
        "0.535", 
        "0.524", 
        "0.525",
        "Max: 3.93\nMin: 0.565\nMedian: 0.819\nAverage: 1.193", 
        "",
        "Max: 0.575\nMin: 0.523\nMedian: 0.55\nAverage: 0.548", 
        "",
        "Max: 0.566\nMin: 0.515\nMedian: 0.542\nAverage: 0.543", 
        "",
        "0.529", 
        "Max: 0.562\nMin: 0.483\nMedian: 0.525\nAverage: 0.523", 
        "", 
        ".561"
    ],
    "Dataset": [
        "All Teams", 
        "All Teams", 
        "All Teams", 
        "All Teams", 
        "All Teams", 
        "All Teams",
        "Single Team", 
        "", 
        "Single Team", 
        "", 
        "Single Team", 
        "", 
        "All Teams", 
        "Single Team", 
        "", 
        "All Teams"
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Apply styling
styled_df = df.style.set_properties(**{
    'background-color': 'lightblue',
    'color': 'black',
    'border-color': 'white',
    'border-width': '1px',
    'border-style': 'solid',
    'text-align': 'left',
    'font-size': '14pt'
}).set_table_styles([{
    'selector': 'th',
    'props': [('background-color', 'navy'), ('color', 'white')]
}])

# Display the styled DataFrame
print(styled_df)