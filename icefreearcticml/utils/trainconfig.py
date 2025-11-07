from pandas import DataFrame
from icefreearcticml.utils import get_datetime_index

class TrainConfig:

    def __init__(
        self,
        y_var: str,
        x_vars: list[str],
        train_split: float,
        model_name: str,
        **kwargs: dict,
    ) -> None:
        self.y_var = y_var
        self.x_vars = x_vars
        self.train_split = train_split
        self.model_name = model_name
        self.members_for_model = None
        self._set_all_vars()
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _set_all_vars(self) -> list[str]:
        if isinstance(self.y_var, list):
            self.all_vars = [*self.y_var, *self.x_vars]
        else:
            self.all_vars = [self.y_var, *self.x_vars]

    def get_emul_test_df(self, model_data: dict) -> DataFrame:
        return filter_by_years(
            model_data[self.y_var][self.model_name],
            self.emul_start, self.emul_end
        )[self.test_members]

    def set_all_data(self, model_data: dict, data: dict | None = None) -> None:
        merged = None
        for var in self.all_vars:
            to_filter = model_data[var][self.model_name].fillna(0)
            if self.members_for_model is not None:
                to_filter = to_filter[self.members_for_model]
            res = get_melt(to_filter, var)
            if merged is None:
                merged = res
            elif var not in merged.columns:
                merged = merged.merge(res, on=["member", "time"])

        merged["time"] = merged["time"].dt.year
        self.all_data = merged
        self.set_simul_datetime_index()
        self.set_train_test_members()

    def set_members_for_model(self, members_for_model: list) -> None:
        self.members_for_model = members_for_model

    def set_simul_datetime_index(self) -> DatetimeIndex:
        self.simul_start = self.all_data["time"].values[0]
        self.simul_end = self.all_data["time"].values[-1]
        start = self.simul_start
        if hasattr(self, "max_encoder_length"):
            start += self.max_encoder_length
        self.simul_index = get_datetime_index(start, self.simul_end)

    def set_emul_data(self) -> None:
        self.emul_df = self.all_data[self.all_data["time"] > (self.simul_end - self.max_encoder_length)]
        self.emul_train = self.emul_df[self.emul_df["member"].isin(self.train_members)]
        self.emul_test = self.emul_df[self.emul_df["member"].isin(self.test_members)]

    def set_simul_data(self) -> None:
        self.simul_df = self.all_data[self.all_data["time"] <= self.simul_end]
        self.simul_train = self.simul_df[self.simul_df["member"].isin(self.train_members)]
        self.simul_test = self.simul_df[self.simul_df["member"].isin(self.test_members)]

    def get_simul_test_df(self) -> DataFrame:
        y_test_simul = self.all_data[self.all_data["time"] >= (self.simul_start + self.max_encoder_length)]
        y_test_simul = y_test_simul.set_index(["time", "member"])[self.y_var].unstack()[self.test_members]
        y_test_simul.index = self.simul_index
        return y_test_simul

    def get_test_df(self) -> DataFrame:
        y_test_simul = self.all_data[self.all_data["time"] >= (self.simul_start + self.max_encoder_length)]
        y_test_simul = y_test_simul.set_index(["time", "member"])[self.y_var].unstack()[self.test_members]
        y_test_simul.index = self.simul_index
        return y_test_simul

    def set_train_test_members(self) -> None:
        self.n_time_steps = len(self.all_data["time"].unique())
        self.n_members = len(self.all_data["member"].unique())
        train_members, test_members = get_train_test_ensembles(self.n_members, self.train_split)
        self.test_members = self.all_data["member"].unique()[test_members]
        self.train_members = self.all_data["member"].unique()[train_members]

    @property
    def train_data(self) -> DataFrame:
        return self.all_data[self.all_data["member"].isin(self.train_members)]

    @property
    def test_data(self) -> DataFrame:
        return self.all_data[self.all_data["member"].isin(self.test_members)]