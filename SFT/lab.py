def parse_lab_events(lab_events_df_sf, _id):
    filtered_lab_events = lab_events_df_sf[lab_events_df_sf["hadm_id"] == _id]
    le, ref_r_low, ref_r_up = {}, {}, {}
    if not filtered_lab_events.empty:
        sorted_df = filtered_lab_events.sort_values(by="charttime", ascending=True)
        unique_lab_events_df = sorted_df.drop_duplicates(subset="itemid", keep="first")
        le = unique_lab_events_df.set_index("itemid")["valuestr"].to_dict()
        ref_r_low = unique_lab_events_df.set_index("itemid")[
            "ref_range_lower"
        ].to_dict()
        ref_r_up = unique_lab_events_df.set_index("itemid")["ref_range_upper"].to_dict()
    return le, ref_r_low, ref_r_up

def parse_microbio(microbio_df_sf, _id):
    filtered_microbio_df = microbio_df_sf[microbio_df_sf["hadm_id"] == _id]
    microbio = {}
    microbio_spec = {}

    # If there are multiple positive bacteria for an itemid, we want to merge them
    def return_value_string(group):
        first_row = group.iloc[0]
        if pd.isna(first_row.org_itemid):
            val_str = first_row.valuestr
        else:
            unique_org_itemid_values = group.dropna(subset=["org_itemid"])[
                "valuestr"
            ].unique()
            val_str = ", ".join(unique_org_itemid_values)
        return pd.Series([val_str, first_row.spec_itemid])

    if not filtered_microbio_df.empty:
        result = (
            filtered_microbio_df.groupby(["test_itemid", "charttime"])
            .apply(return_value_string)
            .reset_index()
        )
        result.columns = ["test_itemid", "charttime", "valuestr", "spec_itemid"]

        # Sort and drop duplicates, creating new DataFrames
        sorted_df = result.sort_values(by="charttime", ascending=True)
        unique_microbio_df = sorted_df.drop_duplicates(
            subset="test_itemid", keep="first"
        )
        microbio = unique_microbio_df.set_index("test_itemid")["valuestr"].to_dict()
        microbio_spec = unique_microbio_df.set_index("test_itemid")[
            "spec_itemid"
        ].to_dict()

    return microbio, microbio_spec