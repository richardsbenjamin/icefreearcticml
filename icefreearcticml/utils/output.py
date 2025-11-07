



class Output:

    def __init__(
        self,
        y_test_simul: DataFrame,
        y_pred_simul: DataFrame,
        train_config: TrainConfig,
        model_res: dict,
    ) -> None:
        self.y_test_simul = y_test_simul
        self.y_pred_simul = y_pred_simul
        self.train_config = train_config
        self.model_res = model_res
        self._simul_resids = None
        self._emul_resids = None

    @property
    def simul_resids(self) -> None:
        if self._simul_resids is None:
            self._simul_resids = (
                (self.y_test_simul - self.y_pred_simul)
                .reset_index(names="time")
                .melt(id_vars=["time"], var_name="member", value_name="resid")
            )
        return self._simul_resids

    @property
    def emul_resids(self) -> None:
        if self._emul_resids is None:
            self._emul_resids = (
                (self.y_test_emul - self.y_pred_emul)
                .reset_index(names="time")
                .melt(id_vars=["time"], var_name="member", value_name="resid")
            )
        return self._emul_resids