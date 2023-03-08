import pandas
from pandas import DataFrame, ExcelWriter
import os
import json

def get_string(n):
    n= n- 65
    str = ''
    while n > 0:
        rem = n % 26
        if rem == 0:
            str += 'Z'
            n = int(n / 26) - 1
        else:
            str += chr(rem - 1 + 65)
            n = int(n / 26)
    return str[::-1]

FILE_PATH = "./results/"
filenames = []
files_to_read = {}
dataframe = {}
directory = FILE_PATH
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".json"):
        filenames.append(filename)
filenames = sorted(filenames)
files_to_read["SPIRAL"] = {}
files_to_read["LBSPIRAL"] = {}
types = []
nodes = []
operations = []
rounds = []
for file in filenames:
    filename = os.fsdecode(file)
    type = filename.split('_')[0]
    types.append(type)
    operation_count = filename.split("_O")[1].split('_P')[0]
    operations.append(operation_count)
    node_count = filename.split("_P")[1].split("_R")[0]
    nodes.append(node_count)
    if node_count in files_to_read[type]:
        if operation_count in files_to_read[type][node_count]:
            files_to_read[type][node_count][operation_count].append(filename)
        else:
            files_to_read[type][node_count][operation_count] = [filename]
    else:
        op_wise = {operation_count: [filename]}
        files_to_read[type][node_count] = op_wise
    print(os.path.join(directory, filename))

# print(files_to_read)

json_result = {}
for type in types:
    json_result[type] = {}

for type in files_to_read:
    for node_count in files_to_read[type]:
        for op_count in files_to_read[type][node_count]:
            for res in files_to_read[type][node_count][op_count]:
                filename = './results/' + res
                # res_json = pandas.read_json(filename,index=[0])
                data = json.load(open(filename))

                if node_count in json_result[type]:
                    if op_count in json_result[type][node_count]:
                        json_result[type][node_count][op_count].append(data)
                    else:
                        json_result[type][node_count][op_count] = [data]
                else:
                    op_wise = {op_count: [data]}
                    json_result[type][node_count] = op_wise

print(json_result)

dataframe = {}
dataframe_map_processing_load = {}
dataframe_map_communication_cost = {}

for type in json_result:
    for node_count in json_result[type]:
        if node_count not in json_result:
            dataframe_map_processing_load[node_count] = {}
            for op_count in operations:
                dataframe_map_processing_load[node_count][op_count] = {}

for type in json_result:
    for node_count in json_result[type]:
        if node_count not in json_result:
            dataframe_map_communication_cost[node_count] = {}
            for op_count in operations:
                dataframe_map_communication_cost[node_count][op_count] = {}

# todo change reps to a variable
reps = 1


processing_load_result = {}
columns = ["Round 1", "Round 2", "Round 3"]
for type in json_result:
    for node_count in json_result[type]:

        for op_count in json_result[type][node_count]:
            optimal_values = {}
            values = {}

            round = 0
            optimal_values["node_id"] = []
            values["node_id"] = []
            for i in range(0, int(node_count)):
                optimal_values["node_id"].append(i + 1)
                values["node_id"].append(i + 1)

            for res in json_result[type][node_count][op_count]:
                round += 1
                optimal_values[round] = []
                values[round] = []
                for load in res['PROCESSING_LOAD_OPTIMAL']:
                    optimal_values[round].append(res['PROCESSING_LOAD_OPTIMAL'][load])
                    # optimal_values[round].append(load)
                for load in res['PROCESSING_LOAD']:
                    values[round].append(res['PROCESSING_LOAD'][load])

            dataframe_map_processing_load[node_count][op_count]['OPTIMAL'] = optimal_values
            if type not in dataframe_map_processing_load[node_count][op_count]:
                dataframe_map_processing_load[node_count][op_count][type] = values
print(dataframe_map_processing_load)

communication_cost_result = {}
for type in json_result:
    for node_count in json_result[type]:

        for op_count in json_result[type][node_count]:
            optimal_values = {}
            values = {}

            round = 0
            optimal_values["node_id"] = []
            values["node_id"] = []
            round_numbers=[]
            for i in range(0, int(node_count)):
                optimal_values["node_id"].append(i + 1)
                values["node_id"].append(i + 1)
            for i in range(0, reps):
                round_numbers.append("Round "+str(i + 1))

            optimal_values[round] = []
            for res in json_result[type][node_count][op_count]:
                dataframe_map_communication_cost[node_count][op_count]["ROUNDS"] = round_numbers

                round += 1
                # dataframe_map_communication_cost[node_count][op_count].setdefault('COST_OPTIMAL', []).append(res['COST_OPTIMAL'])
                if type == "LBSPIRAL":
                    dataframe_map_communication_cost[node_count][op_count].setdefault('COST_OPTIMAL', []).append(
                        res['COST_OPTIMAL'])

                    dataframe_map_communication_cost[node_count][op_count].setdefault('LB_SPIRAL_COST_WITHOUT_INFORM', []).append(
                        res['COST_WITHOUT_INFORM'])
                    dataframe_map_communication_cost[node_count][op_count].setdefault('LB_SPRIAL_COST_INFORM',[]).append(
                        res['COST_INFORM'])
                    dataframe_map_communication_cost[node_count][op_count].setdefault('LB_SPRIAL_COST',[]).append(
                        res['COST'])
                elif type == "SPIRAL":
                    dataframe_map_communication_cost[node_count][op_count].setdefault('SPRIAL_COST', []).append(
                        res['COST'])
print (dataframe_map_communication_cost)
with ExcelWriter('output.xlsx') as writer:
    workbook = writer.book

    # write processing load
    for node_count in dataframe_map_processing_load:
        startrow = 1
        startcol = 0
        for op_count in dataframe_map_processing_load[node_count]:

            sheet_name = (str(node_count) + " nodes, " + op_count + " operations")
            worksheet = workbook.add_worksheet(sheet_name)

            writer.sheets[sheet_name] = worksheet

            col_char = 0
            for data in dataframe_map_processing_load[node_count][op_count]:
                worksheet.write(0, col_char + reps + 2, data)
                df = pandas.DataFrame.from_dict(dataframe_map_processing_load[node_count][op_count][data])

                for x in range(0, int(node_count)):
                    worksheet.write_formula(2 + x,
                                            col_char + reps + 2,
                                            '=AVERAGE(' + get_string(65 + col_char + 3) + str(
                                                x + 3) + ':' + get_string(65 + col_char + 3 + reps - 1) + str(
                                                x + 3) + ')')
                df.to_excel(writer, sheet_name=sheet_name, startrow=startrow, startcol=col_char)

                col_char += reps + 5
            worksheet.write_formula(1, col_char + 2,
                                    '=(' + get_string(65 + reps + 3) + str(1) + ')')
            worksheet.write_formula(1, col_char + 3,
                                    '=(' + get_string(65 + (reps + 3) * 2 + 2) + str(1) + ')')
            worksheet.write_formula(1, col_char + 4,
                                    '=(' + get_string(65 + (reps + 3) * 3 + 2 + 2) + str(1) + ')')
            for x in range(0, int(node_count)):
                worksheet.write_formula(2 + x,
                                        col_char + 2,
                                        '=(' + get_string(65 + reps + 3) + str(x + 3) + ')')

                worksheet.write_formula(2 + x,
                                        col_char + 3,
                                        '=(' + get_string(65 + (reps + 3) * 2 + 2) + str(x + 3) + ')')

                worksheet.write_formula(2 + x,
                                        col_char + 4,
                                        '=(' + get_string(65 + (reps + 3) * 3 + 2 + 2) + str(x + 3) + ')')

    # write communication cost
    sheet_name = "Communication cost"
    worksheet = workbook.add_worksheet(sheet_name)
    writer.sheets[sheet_name] = worksheet
    startrow = 0
    for node_count in dataframe_map_communication_cost:
        startcol = 0
        worksheet.write(startrow, startcol, node_count)
        # for varying operation count
        col_char = 0

        for op_count in dataframe_map_communication_cost[node_count]:
            worksheet.write(startrow, startcol + 1, op_count)

            df = pandas.DataFrame.from_dict(dataframe_map_communication_cost[node_count][op_count])
            for x in range(1, len(dataframe_map_communication_cost[node_count][op_count])):
                worksheet.write_formula(startrow + reps + 2,
                                    col_char + 2 ,
                                    '=AVERAGE(' + get_string(65 + col_char + 3) + str(
                                        startrow + 3) + ':' + get_string(65 + col_char + 3) + str(
                                        startrow + 2 + reps) + ')')
                col_char += 1
            col_char += 5
            df.to_excel(writer, sheet_name=sheet_name, startrow=startrow+1, startcol=startcol)
            startcol = startcol + 10
        startrow = startrow + reps + 5
