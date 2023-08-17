'''
Mapping of data from one tabular representation to another.
The idea is that in the source representation, each 'case' represents one row,
containing information about different laboratory events.
With laboratory event we mean here lab name, method, dates, and results
Each row consists for the source of one sample.
In the target representation, each laboratory event is its own row,
so that sample ids can be repeated (a sample can be analyzed by different labs or different methods).
'''

import os
import pandas as pd
import json
from state_table_to_map_demo import sample_events


def read_excel_files(directory):
    column_names_list = []
    
    # Get all files in the directory
    files = os.listdir(directory)
    column_names_dict = {}
    # Iterate through each file
    for file in files:
        print("Working on file", file)
        if (file.endswith(".xlsx") or file.endswith(".xls")) and not file.startswith('~'):
            # Construct the file path
            file_path = os.path.join(directory, file)
            # Read the Excel file
            df = pd.read_excel(file_path)
            # Get the column names and store them in a list
            column_names = df.columns.tolist()
            base = os.path.splitext(os.path.basename(file_path))[0]
            # Append the column names to the main list
            # column_names_list.append(column_names)
            column_names_dict[base] = column_names
    return column_names_dict



def test_read_excel_files(dir='../data/raw/fbd_state'):
    result = read_excel_files(directory=dir)
    print(result)


def test_read_excel_file(file='../data/raw/fbd_state/ExploratoryandSpecialProgramPoultry_SampleDataset.xlsx'):
    result = pd.read_excel(file)
    print(result)


def table_to_map(df, source='Source element (State)', target='Target Element (FDA)'):
    '''
    Generates mapping dictionary from tabular mapping description
    '''
    df = df.copy(deep=True).reset_index(drop=True)
    map = {}
    for i in range(len(df)):
        t = df.at[i, target] # target column
        s = df.at[i, source] # source column
        if t in map:
            continue
        # count occurrences
        count = df[target].value_counts()[t]
        if count == 1:
            map[t] = s
        else:
            flags = df[target] == t
            values = df[flags][source].tolist()
            map[t] = values
    return map


def test_table_to_map(file='../data/raw/Mapping State Source to FDA Target v2.0.xlsx',
    outfile='tmp_test_table_to_map_outdesc.py'):
    df = pd.read_excel(file)
    result = table_to_map(df)
    print(result)
    # Open a file in write mode
    if outfile is not None:
        with open(outfile, "w") as file:
            print("writing to outpuf file", outfile)
            # Write the dictionary to file with pretty formatting
            json.dump(result, file, indent=4)


def multiresult_to_multirow(df, map, positive=['positive'],
    delim=' @ ', delim2=':'):
    '''
    Performs actuall mapping and creates output data frame with mapping information.
    '''
    df = df.copy(deep=True).reset_index(drop=True)
    kinds = []
    results = []
    remarks=[]
    valuemap = {}
    events = map['events']
    labsample = map['labsample']
    print(list(df.columns))
    for eventname in events.keys():
        print(eventname)
        items = events[eventname]
        resultcol = items['Result']
        print('result column name', resultcol)
        # kinds.append(resultcol)
        if resultcol not in df.columns:
            print("Could not find result column:", df.columns)
        assert resultcol in df.columns
        print(df[resultcol])
        for resultrow in range(len(df)):
            result = str(df.at[resultrow, resultcol])
            if result is None or result.lower() not in positive:
                print("skipping result:",resultrow, resultcol, result)
                continue
            for item in items.keys():
                print('item:', item)
                assert item in items
                mitem = items[item]
                print('mapped item:', mitem)
                value = mitem
                if isinstance(mitem, list): # list, to be concatenated
                    value = ''
                    for mitem2 in mitem:
                        value2 = mitem2
                        if mitem2 in df.columns:
                            if pd.isna(df.at[resultrow, mitem2]):
                                continue
                            else:
                                value2 = str(df.at[resultrow, mitem2]).replace("\n", "").replace("\r", "")
                        if len(value) > 0:
                            value = value + delim 
                        value = value + mitem2 + delim2 + value2
                else:
                    if mitem in df.columns:
                        value = str(df.at[resultrow, mitem]).replace("\n", "").replace("\r", "")
                print('value:', value)
                if item not in valuemap:
                    valuemap[item] = []
                print(f"Adding item {item}: {value}")
                valuemap[item].append(value)
            remark = ''
            for col in df.columns:
                if col not in items.keys():
                    value = df.at[resultrow,col]
                    if not pd.isna(value):
                        value = str(value)
                        if len(value) > 0:
                            if len(remark) > 0:
                                remark = remark + delim 
                            remark = remark + col + delim2 + value
            for topic in labsample.keys(): # loop over firm, lab, sample etc:
                for item in labsample[topic]:
                    mitem = labsample[topic][item]
                    value = mitem
                    if pd.isna(mitem):
                        value = pd.NA
                    if mitem is not pd.isna(mitem) and mitem in df.columns:
                        value = df.at[resultrow,mitem]
                    if item not in valuemap:
                        valuemap[item] = []
                    print(f"Adding item {item}: {value}")
                    valuemap[item].append(value)                    
            kinds.append(eventname)
            assert result.lower() in positive
            results.append(result)
            remarks.append(remark)
    valuemap['Result'] = results
    valuemap['Kind'] = kinds
    valuemap['Remark'] = remarks
    df2 = pd.DataFrame(data=valuemap)
    return df2


def test_multiresult_to_multirow(infile='../data/raw/fbd_state/Dataset_RawChickenCarcasses_Current.xlsx',
    outfile="tmp_test_multiresult_to_multirow.csv"):
    dat = pd.read_excel(infile, sheet_name='Extracted_03_16_2023',skiprows=5)
    result=multiresult_to_multirow(dat, map=sample_events)
    print(result)
    if outfile is not None:
        result.to_csv(outfile)


if __name__ == '__main__':
    test_multiresult_to_multirow()
    assert False
    test_table_to_map()
    assert False
    test_read_excel_file()

