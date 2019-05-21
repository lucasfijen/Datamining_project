prop_dest_avg_corrected_pos = dict()

for prop in df['prop_id'].unique():
    for dest in df['srch_destination_id'].unique():
        prop_dest_set = df.loc[(df['prop_id'] == prop) & df['srch_destination_id'].isin([dest])]
        prop_dest_avg_corrected_pos[prop+dest]= np.average(prop_dest_set['corrected_position'])

prop_dest_avg_gain = dict()

for prop in df['prop_id'].unique():
    for dest in df['srch_destination_id'].unique():
        prop_dest_set = df.loc[(df['prop_id'] == prop) & df['srch_destination_id'].isin([dest])]
        prop_dest_avg_gain[prop+dest]= np.average(prop_dest_set['non_corrected_total'])
