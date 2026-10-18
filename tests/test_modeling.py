import numpy as np
import pandas as pd
from modeling import prepare_features, split_players, train_model

def dataset():
    return pd.DataFrame({"Player":[f"player{i}" for i in range(30)], "Pos":["FW","MF","DF"]*10,
        "Age":np.arange(20,50,dtype=float), "Min":np.arange(30)*50., "Gls":np.arange(30)%7,
        "Ast":np.arange(30)%4,"Gls_90":np.arange(30)*100.,"Ast_90":np.arange(30)*200.})

def test_outcome_derived_features_are_not_predictors():
    data, features=prepare_features(dataset())
    assert not {"Gls", "Ast", "Gls_90", "Ast_90"}.intersection(features)
    model,_=train_model(data, features+["Gls_90"], "Gls")
    assert "Gls_90" not in model.feature_names_in_

def test_the_same_player_cannot_appear_on_both_sides_of_split():
    data=pd.concat([dataset(),dataset()],ignore_index=True)
    train,test=split_players(data)
    assert set(data.iloc[train].Player).isdisjoint(data.iloc[test].Player)

def test_imputation_is_fitted_only_on_training_players():
    data=dataset(); train,test=split_players(data)
    data.loc[test,"Age"]=100000.
    data.loc[train[0],"Age"]=np.nan
    prepared,features=prepare_features(data)
    model,metrics=train_model(prepared,features,"Gls")
    median=model.named_steps['preprocessing'].named_transformers_['numeric'].statistics_[0]
    assert median == data.iloc[train].Age.median()
    assert np.isfinite(metrics['Baseline MAE Test'])

def test_unseen_positions_and_missing_numeric_values_are_supported():
    data,features=prepare_features(dataset());model,_=train_model(data,features,"Ast")
    sample=data.iloc[:1].copy();sample['Pos']='UNSEEN';sample['Age']=np.nan
    assert np.isfinite(model.predict(sample)[0])
