"""
Created on Mon Nov 25 14:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Helper routines for working with SmartSheets

"""

from collections import defaultdict
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

        # Lookups
        self.column_name_to_id = {c.title: c.id for c in self.sheet.columns}

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

    # --- Getters ---
    def get_children_map(self):
        children_map = defaultdict(list)
        idx_lookup = {row.id: idx for idx, row in enumerate(self.sheet.rows)}
        for row in self.sheet.rows:
            if row.parent_id:
                parent_idx = idx_lookup[row.parent_id]
                child_idx = idx_lookup[row.id]
                children_map[parent_idx].append(child_idx)
        return children_map

    def get_rows_in_column_with(self, column_name, row_value):
        row_idxs = list()
        col_id = self.column_name_to_id[column_name]
        for idx, row in enumerate(self.sheet.rows):
            cell = next((c for c in row.cells if c.column_id == col_id), None)
            value = cell.display_value or cell.value
            if isinstance(value, str):
                if value.lower() == row_value.lower():
                    row_idxs.append(idx)
        return row_idxs

    def get_value(self, row_idx, column_name):
        row = self.sheet.rows[row_idx]
        col_id = self.column_name_to_id[column_name]
        cell = next((c for c in row.cells if c.column_id == col_id), None)
        return cell.display_value or cell.value

    # --- Miscellaneous ---
    def to_dataframe(self):
        # Extract column titles
        columns = list(self.column_name_to_id.keys())

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
def extract_somas(smartsheet_client, microscope="ExaSPIM", status=None):
    # Extract rows by microscope and brain_id
    idxs = smartsheet_client.get_rows_in_column_with("Collection", microscope)
    children_map = smartsheet_client.get_children_map()
    children_map = {k: v for k, v in children_map.items() if k in idxs}

    # Extract soma coordinates
    soma_locations = dict()
    for parent_idx, child_idxs in children_map.items():
        brain_id = smartsheet_client.get_value(parent_idx, "ID")
        xyz_list = get_coordinates(smartsheet_client, child_idxs, status)
        if len(xyz_list) > 0:
            soma_locations[brain_id] = xyz_list
    return soma_locations


def get_coordinates(smartsheet_client, row_idxs, status=None):
    # Initializations
    status_column_id = smartsheet_client.column_name_to_id["Status 1"]
    coord_column_id = smartsheet_client.column_name_to_id["Horta Coordinates"]

    # Parse rows
    xyz_list = list()
    for idx in row_idxs:
        # Search row
        soma_status, soma_xyz = None, None
        for cell in smartsheet_client.sheet.rows[idx].cells:
            if cell.column_id == status_column_id:
                soma_status = cell.display_value or cell.value
            elif cell.column_id == coord_column_id:
                soma_xyz = read_xyz(cell.display_value or cell.value)

        # Process result
        if soma_status == status and soma_xyz:
            xyz_list.append(soma_xyz)
        elif status is None and soma_xyz:
            xyz_list.append(soma_xyz)
    return xyz_list


def read_xyz(xyz_str):
    try:
        return ast.literal_eval(xyz_str)
    except:
        return None


# --- ExM Dataset Summary Utils ---
def update_soma_detection(brain_id, access_token):
    # Initialize client
    sheet_name = "ExM Dataset Summary"
    client = SmartSheetClient(access_token, sheet_name)
    column_map = {col.title: col.id for col in client.sheet.columns}

    # Update SmartSheet
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
