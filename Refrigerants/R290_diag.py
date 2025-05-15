def phdiagram_R290():
    import pandas as pd
    import os
    import pylab
    import matplotlib.pyplot as plt
    import seaborn as sns


    os.chdir('/Users/pnmsharma/Documents/Python/Refrigeration/')
 

    filepath='/Users/pnmsharma/Documents/Python/Refrigeration/R290.csv'
    R290_saturated_input_data=pd.read_csv(filepath, encoding='cp1252')
    R290_saturated_input_data['DRUCK']=R290_saturated_input_data['DRUCK']/100000
    R290_saturated_input_data['Enthalpy_Quality0.1']=R290_saturated_input_data['ENTHA0']+0.1*(R290_saturated_input_data['ENTHA1']-R290_saturated_input_data['ENTHA0'])
    R290_saturated_input_data['Enthalpy_Quality0.2']=R290_saturated_input_data['ENTHA0']+0.2*(R290_saturated_input_data['ENTHA1']-R290_saturated_input_data['ENTHA0'])
    R290_saturated_input_data['Enthalpy_Quality0.3']=R290_saturated_input_data['ENTHA0']+0.3*(R290_saturated_input_data['ENTHA1']-R290_saturated_input_data['ENTHA0'])
    R290_saturated_input_data['Enthalpy_Quality0.4']=R290_saturated_input_data['ENTHA0']+0.4*(R290_saturated_input_data['ENTHA1']-R290_saturated_input_data['ENTHA0'])
    R290_saturated_input_data['Enthalpy_Quality0.5']=R290_saturated_input_data['ENTHA0']+0.5*(R290_saturated_input_data['ENTHA1']-R290_saturated_input_data['ENTHA0'])
    R290_saturated_input_data['Enthalpy_Quality0.6']=R290_saturated_input_data['ENTHA0']+0.6*(R290_saturated_input_data['ENTHA1']-R290_saturated_input_data['ENTHA0'])
    R290_saturated_input_data['Enthalpy_Quality0.7']=R290_saturated_input_data['ENTHA0']+0.7*(R290_saturated_input_data['ENTHA1']-R290_saturated_input_data['ENTHA0'])
    R290_saturated_input_data['Enthalpy_Quality0.8']=R290_saturated_input_data['ENTHA0']+0.8*(R290_saturated_input_data['ENTHA1']-R290_saturated_input_data['ENTHA0'])
    R290_saturated_input_data['Enthalpy_Quality0.9']=R290_saturated_input_data['ENTHA0']+0.9*(R290_saturated_input_data['ENTHA1']-R290_saturated_input_data['ENTHA0'])

    max_enthal_index = R290_saturated_input_data['Enthalpy_Quality0.1'].idxmax()
    R290_saturated_input_data_1_1 = R290_saturated_input_data.loc[:max_enthal_index]
    R290_saturated_input_data_1_2 = R290_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R290_saturated_input_data['Enthalpy_Quality0.2'].idxmax()
    R290_saturated_input_data_2_1 = R290_saturated_input_data.loc[:max_enthal_index]
    R290_saturated_input_data_2_2 = R290_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R290_saturated_input_data['Enthalpy_Quality0.3'].idxmax()
    R290_saturated_input_data_3_1 = R290_saturated_input_data.loc[:max_enthal_index]
    R290_saturated_input_data_3_2 = R290_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R290_saturated_input_data['Enthalpy_Quality0.4'].idxmax()
    R290_saturated_input_data_4_1 = R290_saturated_input_data.loc[:max_enthal_index]
    R290_saturated_input_data_4_2 = R290_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R290_saturated_input_data['Enthalpy_Quality0.5'].idxmax()
    R290_saturated_input_data_5_1 = R290_saturated_input_data.loc[:max_enthal_index]
    R290_saturated_input_data_5_2 = R290_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R290_saturated_input_data['Enthalpy_Quality0.6'].idxmax()
    R290_saturated_input_data_6_1 = R290_saturated_input_data.loc[:max_enthal_index]
    R290_saturated_input_data_6_2 = R290_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R290_saturated_input_data['Enthalpy_Quality0.7'].idxmax()
    R290_saturated_input_data_7_1 = R290_saturated_input_data.loc[:max_enthal_index]
    R290_saturated_input_data_7_2 = R290_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R290_saturated_input_data['Enthalpy_Quality0.8'].idxmax()
    R290_saturated_input_data_8_1 = R290_saturated_input_data.loc[:max_enthal_index]
    R290_saturated_input_data_8_2 = R290_saturated_input_data.loc[max_enthal_index:]
    max_enthal_index = R290_saturated_input_data['Enthalpy_Quality0.9'].idxmax()
    R290_saturated_input_data_9_1 = R290_saturated_input_data.loc[:max_enthal_index]
    R290_saturated_input_data_9_2 = R290_saturated_input_data.loc[max_enthal_index:]



    series_1 = pd.Series(R290_saturated_input_data['ENTHA0'])
    series_2 = pd.Series(R290_saturated_input_data['ENTHA1'])
    series_2=series_2[::-1]

    Enthal = pd.concat([series_1, series_2], ignore_index=True)

    series_a = pd.Series(R290_saturated_input_data['DRUCK'])
    series_b = pd.Series(R290_saturated_input_data['DRUCK'])
    series_b=series_b[::-1]
    DRUCK = pd.concat([series_a, series_b], ignore_index=True)
    concatenated_df = pd.concat([Enthal, DRUCK], axis=1)
    concatenated_df.rename(columns={0:'Enthalpy','DRUCK': 'Pressure'}, inplace=True)


    max_enthal_index = concatenated_df['Enthalpy'].idxmax()
    df1 = concatenated_df.loc[:max_enthal_index]
    df2 = concatenated_df.loc[max_enthal_index:]

    fig,ax = plt.subplots()

    sns.lineplot(data=R290_saturated_input_data_1_1, x='Enthalpy_Quality0.1', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R290_saturated_input_data_2_1, x='Enthalpy_Quality0.2', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R290_saturated_input_data_3_1, x='Enthalpy_Quality0.3', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R290_saturated_input_data_4_1, x='Enthalpy_Quality0.4', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R290_saturated_input_data_5_1, x='Enthalpy_Quality0.5', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R290_saturated_input_data_6_1, x='Enthalpy_Quality0.6', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R290_saturated_input_data_7_1, x='Enthalpy_Quality0.7', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R290_saturated_input_data_8_1, x='Enthalpy_Quality0.8', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R290_saturated_input_data_9_1, x='Enthalpy_Quality0.9', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R290_saturated_input_data_1_2, x='Enthalpy_Quality0.1', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R290_saturated_input_data_2_2, x='Enthalpy_Quality0.2', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R290_saturated_input_data_3_2, x='Enthalpy_Quality0.3', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R290_saturated_input_data_4_2, x='Enthalpy_Quality0.4', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R290_saturated_input_data_5_2, x='Enthalpy_Quality0.5', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R290_saturated_input_data_6_2, x='Enthalpy_Quality0.6', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R290_saturated_input_data_7_2, x='Enthalpy_Quality0.7', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R290_saturated_input_data_8_2, x='Enthalpy_Quality0.8', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')
    sns.lineplot(data=R290_saturated_input_data_9_2, x='Enthalpy_Quality0.9', y='DRUCK',linewidth=0.4,color='lightblue', linestyle='dashed')

    sns.lineplot(data=df1, x='Enthalpy', y='Pressure',linewidth=0.5,color='black')
    sns.lineplot(data=df2, x='Enthalpy', y='Pressure',linewidth=0.5,color='black')


    #Set plot title and axis labels
    plt.title('R290',fontsize=8)
    plt.xlabel('Enthalpy [kJ/kg]',fontsize=5)
    plt.ylabel('Pressure [barA]',fontsize=5)
    plt.yscale('log') 
    plt.xlim(50, 700)  # Set x-axis limits
    plt.ylim(0.9, 200)  # Set x-axis limits
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)

    #return fig,ax

    # Display the plot
    plt.show()


phdiagram_R290()





   



