import os
import json
import pandas as pd
import argparse
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

topic_plot_names = {
    'other': 'Other',
    'stem': 'STEM',
    'humanities': 'Humanities',
    'social_sciences': 'Social Sciences',
    'score': 'Overall'
}

def compare_per_language(input_folder, output_folder):
    data = {}
    
    # Iterate over all files in the folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            file_path = os.path.join(input_folder, filename)
            with open(file_path, 'r') as file:
                content = json.load(file)
                # Extract language and model from filename
                parts = filename.split('_')
                language = parts[1]
                # Remove the extension
                model = parts[2][:-5]  
                
                if language not in data:
                    data[language] = []
                
                # Add model data to the language
                data[language].append({
                    'model': model,
                    'other': content['other'],
                    'other:std': content['other:std'],
                    'stem': content['stem'],
                    'stem:std': content['stem:std'],
                    'humanities': content['humanities'],
                    'humanities:std': content['humanities:std'],
                    'social_sciences': content['social_sciences'],
                    'social_sciences:std': content['social_sciences:std'],
                    'score': content['score'],
                    'score:std': content['score:std']
                })
    
    # Create a DataFrame for each language and save to CSV
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    data_per_lang = {}
    for language, models_data in tqdm(data.items(), desc='Processing results per language', leave=False):
        df = pd.DataFrame(models_data)
        
        # Cast numerical columns to float
        numerical_columns = ['other', 'other:std', 'stem', 'stem:std', 'humanities', 'humanities:std', 'social_sciences', 'social_sciences:std', 'score', 'score:std']
        df[numerical_columns] = df[numerical_columns].astype(float)
        
        output_file = os.path.join(output_folder, f"{language}_comparison.csv")
        df.to_csv(output_file, index=False)

        data_per_lang[language] = df

    return data_per_lang 
        

def plot_per_language(data, output_folder):
        
    for language, df in tqdm(data.items(), desc='Plotting results per language', leave=False):
        # Generate plots
        plot_file = os.path.join(output_folder, f"{language}_comparison.pdf")
        plt.figure(figsize=(18, 8)) 
        
        # Prepare data for plotting
        topics = ['other', 'stem', 'humanities', 'social_sciences', 'score']
        stds = ['other:std', 'stem:std', 'humanities:std', 'social_sciences:std', 'score:std']
        
        plot_data = pd.DataFrame()
        for topic, std in zip(topics, stds):
            temp_df = df[['model', topic, std]].copy()
            temp_df.columns = ['model', 'score', 'std']
            temp_df['topic'] = topic_plot_names[topic]
            plot_data = pd.concat([plot_data, temp_df])
        
        # Pivot the data for plotting
        plot_data_pivot = plot_data.pivot(index='topic', columns='model', values='score')
        plot_data_pivot_std = plot_data.pivot(index='topic', columns='model', values='std')
        
        ax = plot_data_pivot.plot(kind='bar', yerr=plot_data_pivot_std, capsize=4, error_kw=dict(ecolor='black', lw=1, capsize=4, capthick=1))
        
        #plt.title(f"Comparison of Models for {language}")
        plt.ylabel("Scores")
        plt.xlabel("Topics")
        plt.xticks(rotation=45)
        
        # Move the legend outside the plot
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), ncol=5)
        plt.grid(axis='y', linestyle='--')
        plt.tight_layout(pad=0.5)
        plt.savefig(plot_file, bbox_inches='tight', dpi=300)
        plt.close()


def plot_per_topic(data, output_folder):
    topics = ['other', 'stem', 'humanities', 'social_sciences', 'score']
    
    for topic in tqdm(topics, desc='Plotting results per topic', leave=False):
        comparison_data = []
        
        for language, df in data.items():
            for _, row in df.iterrows():
                comparison_data.append({
                    'language': language,
                    'model': row['model'],
                    'score': row[topic],
                    'std': row[f'{topic}:std']
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Generate plot
        plot_file = os.path.join(output_folder, f"{topic}_comparison_across_languages.pdf")
        plt.figure(figsize=(18, 36))  # Increase the figure height
        
        ax = comparison_df.pivot(index='language', columns='model', values='score').plot(kind='bar', yerr=comparison_df.pivot(index='language', columns='model', values='std').values.T, capsize=4, error_kw=dict(ecolor='black', lw=1, capsize=4, capthick=1))
        
        # Print the latex table 
        #print(50*'=')
        #print(10*'=',topic_plot_names[topic], 10*'=')
        #print(50*'=')
        #print(10*'=',"Scores", 10*'=')
        #print(comparison_df.pivot(index='language', columns='model', values='score').to_latex())
        #print(10*'=',"STD", 10*'=')
        #print(comparison_df.pivot(index='language', columns='model', values='std').to_latex())
        #print()


        #plt.title(f"Comparison of Models for {topic_plot_names[topic]} Across Languages")
        plt.ylabel("Scores")
        plt.xlabel("Languages")
        plt.xticks(rotation=45)
        
        # Move the legend below the x-axis label
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=3)
        plt.grid(axis='y', linestyle='--')
        plt.tight_layout(pad=3.0)  # Increase padding to avoid overlap
        plt.savefig(plot_file, bbox_inches='tight', dpi=300)
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process JSON files and generate comparison DataFrames and plots.')
    parser.add_argument("-i", '--input_folder', type=str, required=True, help='Path to the input folder containing JSON files')
    parser.add_argument("-o", '--output_folder', type=str, required=True, help='Path to the output folder to save CSV files and plots')
    
    args = parser.parse_args()

    data = compare_per_language(args.input_folder, args.output_folder)
    plot_per_language(data, args.output_folder)
    plot_per_topic(data, args.output_folder)