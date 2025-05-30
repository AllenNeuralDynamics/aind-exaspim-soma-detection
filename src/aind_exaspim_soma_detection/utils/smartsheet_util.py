"""
Created on Mon Nov 25 14:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Helper routines for working with SmartSheets

"""

from datetime import datetime

import ast
import pandas as pd

import smartsheet


class SmartSheetClient:
    def __init__(self, access_token, sheet_name):
        # Instance attributes
        self.client = smartsheet.Smartsheet(access_token)
        self.sheet_name = sheet_name
        self.sheet_id = self.find_sheet_id()
        self.sheet = self.client.Sheets.get_sheet(self.sheet_id)

    # --- Lookup Routines ---
    def find_sheet_id(self):
        response = self.client.Sheets.list_sheets()
        for sheet in response.data:
            if sheet.name == self.sheet_name:
                return sheet.id
        raise Exception(f"Sheet Not Found - sheet_name={self.sheet_name}")

    def find_row_id(self, keyword):
        for row in self.sheet.rows:
            for cell in row.cells:
                if cell.display_value == keyword:
                    return row.id
        raise Exception(f"Row Not Found - keyword={keyword}")

    # --- Miscellaneous ---
    def get_dataframe(self):
        # Extract column titles
        columns = [col.title for col in self.sheet.columns]

        # Extract row data
        data = []
        for row in self.sheet.rows:
            row_data = []
            for cell in row.cells:
                val = cell.value if cell.display_value else cell.display_value
                row_data.append(val)
            data.append(row_data)
        return pd.DataFrame(data, columns=columns)

    def update_rows(self, updated_row):
        self.client.Sheets.update_rows(self.sheet_id, [updated_row])


# --- Neuron Reconstruction Utils ---
def extract_somas(df):
    idx = 0
    soma_locations = dict()
    while idx < len(df["Horta Coordinates"]):
        microscope = df["Horta Coordinates"][idx]
        if type(microscope) is str:
            if "spim" in microscope.lower():
                brain_id = str(df["ID"][idx]).split(".")[0]
                xyz_list = extract_somas_by_brain(df, idx + 1)
                if len(xyz_list) > 0:
                    soma_locations[brain_id] = xyz_list
        idx += 1
    return soma_locations


def extract_somas_by_brain(df, idx):
    xyz_list = list()
    while isinstance(df["Horta Coordinates"][idx], str):
        # Check whether to add idx
        entry = df["Horta Coordinates"][idx]
        is_coord = "[" in entry and "]" in entry
        if is_coord:
            try:
                xyz_list.append(ast.literal_eval(entry))
            except:
                pass

        # Check whether reached last row
        idx += 1
        if idx >= len(df["Horta Coordinates"]):
            break
    return xyz_list


# --- ExM Dataset Summary Utils ---
def update_soma_detection(client, brain_id):
    column_map = {col.title: col.id for col in client.sheet.columns}
    updated_row = smartsheet.models.Row()
    updated_row.id = client.find_row_id(brain_id)
    updated_row.cells.append({
        'column_id': column_map.get('Soma Detection'),
        'value': True,
        'strict': False
    })
    updated_row.cells.append({
        'column_id': column_map.get('Soma Detection Date'),
        'value': datetime.today().strftime("%m/%d/%Y"),
        'strict': False
    })
    client.update_rows(updated_row)
    