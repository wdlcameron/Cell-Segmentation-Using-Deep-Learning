import pandas as pd
import numpy as np


def  create_summary_columns(df):
    
    dataframe = df.copy()
    
    #precolumns are the columns like ROI that aren't important for the export
    precolumns = 3
    
    summary_dataframe = pd.DataFrame([])
    
    timepoints = max(dataframe[precolumns:].columns, key = lambda x: x[0])[0]
    print (timepoints)
    
    
    channels = max(dataframe[precolumns:].columns, key = lambda x:x[1])[1]
    print (channels)
    
    
    
    statistics = {"Mean": np.mean, 
                  "Stdev": np.std,
                  "Count": len, "StdErr": lambda x: np.std(x)/np.sqrt(len(x))}
    
    
    
    for t in range (timepoints+1):
        
        series = pd.Series([t], index = [(" "," ","Time")])
        
        for func in statistics:
            temp_dataframe = dataframe.drop(['ROI'], axis = 1, level = 2)         
            new_series = temp_dataframe.loc[0:, pd.IndexSlice[t,0:,:]].apply(statistics[func])
            new_series.index = [(func, i[1], i[2]) for i in new_series.index]
            series = series.append(new_series)
        
        
        columns = series.index
        
        #print (series)
        summary_dataframe = summary_dataframe.append(series, ignore_index = True, sort = False)
        
    
    
    summary_dataframe.columns = pd.MultiIndex.from_tuples(summary_dataframe.columns, names = ['Stat', 'Channel', 'Para'])
    

    
    #print (df.columns)
    #print (summary_dataframe)
    return summary_dataframe


def create_cell_comparisons(df):
    dataframe = df.copy()
    
    #image_type_array = Parameters.image_type_array
    
    #image_type_bool = [image_type_array[i] == "Anisotropy" for i in range(len(image_type_array))]
    
    relative_values = False
    AniChannel = "AniAvg"
    IntChannel = "Intensity"
    
    #print (dataframe)
    
    statistics = {"Mean": np.mean, 
                  "Stdev": np.std,
                  "Count": len, "StdErr": lambda x: np.std(x)/np.sqrt(len(x))}


    channels = {j for (i,j,k) in df if (i ==0 and j>=0)}
    
    
    cell_cell_comparisons = {}
    print ()

    analysis_channels = {AniChannel, IntChannel}
    
    for channel in channels:
        #print (channel, image_type)

        properties = {k for (i,j,k) in df if (i==0 and j == channel)}

        if analysis_channels&properties:
            
            if AniChannel in properties:
                channel_name = AniChannel
                image_type = 'Anisotropy'
                
            elif IntChannel in properties:
                channel_name = IntChannel
                image_type = 'Intensity'

            else:
                print ('Error, neither Intensity or Anisotropy Found')
                
            
            """Note, there is a bug here where the channel cannot be found if no ROIS were written to that channel"""
            channel_subarray = dataframe.loc[:, pd.IndexSlice[:, channel, channel_name]]
            channel_subarray = channel_subarray.dropna(how = 'any')
            
            
            
            #print (image_type, (0, channel, channel_name))
            first_channel = channel_subarray.loc[:, (0, channel, channel_name)].copy()
            
            if relative_values == True:        
                for columns in channel_subarray.columns:
                    channel_subarray.at[:, columns] = channel_subarray.loc[:, columns]/first_channel
                
            for functions in statistics:
                channel_subarray[functions] = channel_subarray.apply(statistics[functions], axis = 1)

            
            cell_cell_comparisons.update({f"{image_type}_Channel_{channel}": channel_subarray})
                
    
    
    return (cell_cell_comparisons)

