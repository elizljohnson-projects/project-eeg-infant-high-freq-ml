import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, 
                              confusion_matrix, roc_auc_score, roc_curve)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneGroupOut

def classify_wake_sleep(df, channel = 'fz_hfb', seed = 325):
    """
    Classify wake vs. sleep using single channel HFB data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe with columns: subject_id, condition, fz_hfb, p3_hfb
    channel : str
        Channel to use for classification ('fz_hfb' or 'p3_hfb')
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    results : dict
        Dictionary containing pipeline, predictions, and evaluation metrics
    """   
    # remove rows with missing data for this channel
    df_clean = df[df[channel].notna()].copy()
    
    print(f'Classification using {channel}')
    print(f'Total samples: {len(df_clean)}')
    print(f'  Awake: {(df_clean.condition == 'awake').sum()}')
    print(f'  Sleep: {(df_clean.condition == 'sleep').sum()}')
    
    # prepare features and labels
    X = df_clean[channel].values.reshape(-1, 1)
    y = (df_clean['condition'] == 'awake').astype(int).values
    
    # split data: 70% train, 30% test, stratified
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.3, random_state = seed, stratify = y
    )
    
    print(f'\nTrain set: {len(X_train)} samples')
    print(f'Test set: {len(X_test)} samples')
    
    # create and train pipeline with balanced weighting 
    pipeline = Pipeline([
        ('classifier', LogisticRegression(class_weight = 'balanced', random_state = seed))
    ])
    
    pipeline.fit(X_train, y_train)
    
    # make predictions
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)
    
    # print results
    print('\nResults:')
    print(f'Accuracy:          {accuracy:.4f}')
    print(f'Balanced accuracy: {balanced_acc:.4f}')
    print(f'F1-score:          {f1:.4f}')
    print(f'ROC-AUC:           {roc_auc:.4f}')
    print('\nConfusion matrix:')
    print('                Predicted')
    print('                Sleep  Awake')
    print(f'Actual  Sleep    {cm[0,0]:4d}   {cm[0,1]:4d}')
    print(f'        Awake    {cm[1,0]:4d}   {cm[1,1]:4d}')
    
    # plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    
    plt.figure(figsize = (4, 4))
    plt.plot(fpr, tpr, linewidth = 2, label = f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth = 1, label = 'Chance (AUC = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(f'ROC curve: {channel}')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # store results
    results = {
        'pipeline': pipeline,
        'channel': channel,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm
    }
    
    return results

def compare_classifiers(df, channel = 'fz_hfb', seed = 325):
    """
    Compare multiple classifiers for wake vs. sleep classification.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe with columns: subject_id, condition, fz_hfb, p3_hfb
    channel : str
        Channel to use for classification ('fz_hfb' or 'p3_hfb')
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    results_df : pandas.DataFrame
        DataFrame with evaluation metrics for each classifier
    """    
    # remove rows with missing data for this channel
    df_clean = df[df[channel].notna()].copy()
    
    print(f'Classifier comparison: {channel}')
    print(f'Total samples: {len(df_clean)}')
    print(f'  Awake: {(df_clean.condition == 'awake').sum()}')
    print(f'  Sleep: {(df_clean.condition == 'sleep').sum()}')
    
    # prepare features and labels
    X = df_clean[channel].values.reshape(-1, 1)
    y = (df_clean['condition'] == 'awake').astype(int).values
    
    # split data: 70% train, 30% test, stratified
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.3, random_state = seed, stratify = y
    )
    
    print(f'\nTrain set: {len(X_train)} samples')
    print(f'Test set: {len(X_test)} samples')
    
    # define classifiers to compare
    classifiers = {
        'Logistic regression': LogisticRegression(class_weight = 'balanced', random_state = seed),
        'SVM (RBF)': SVC(class_weight = 'balanced', kernel = 'rbf', probability = True, random_state = seed),
        'Random forest': RandomForestClassifier(class_weight = 'balanced', n_estimators = 100, random_state = seed),
        'Gradient boosting': GradientBoostingClassifier(n_estimators = 100, random_state = seed)
    }
    
    results = []
    all_fprs = []
    all_tprs = []
    all_aucs = []
    
    # train and evaluate each classifier
    for name, clf in classifiers.items():
        # create pipeline
        pipeline = Pipeline([('classifier', clf)])
        
        # train
        pipeline.fit(X_train, y_train)
        
        # predict
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # store results
        results.append({
            'Classifier': name,
            'Accuracy': accuracy,
            'Balanced accuracy': balanced_acc,
            'F1-score': f1,
            'ROC-AUC': roc_auc
        })
        
        # store ROC curve data
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        all_fprs.append(fpr)
        all_tprs.append(tpr)
        all_aucs.append(roc_auc)
            
    # create results df
    results_df = pd.DataFrame(results)
    
    # print comparison table
    print('\nResults summary:')
    print(results_df.to_string(index = False))
    
    # plot all ROC curves on one plot
    plt.figure(figsize = (6, 6))
    colors = ['blue', 'green', 'red', 'purple']
    
    for i, name in enumerate(classifiers.keys()):
        plt.plot(all_fprs[i], all_tprs[i], linewidth = 2, 
                label = f'{name} (AUC = {all_aucs[i]:.3f})', color = colors[i])
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth = 1, label = 'Chance (AUC = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(f'ROC curves comparison: {channel}')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return results_df

def classify_wake_sleep_standardized(df, channel = 'fz_hfb', n_epochs_per_cond = 20, seed = 325):
    """
    Classify wake vs. sleep using within-subject standardized HFB data with subject-based split.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe with columns: sid, condition, fz_hfb, p3_hfb
    channel : str
        Channel to use for classification ('fz_hfb' or 'p3_hfb')
    n_epochs_per_condition : int
        Number of epochs per condition per subject (default: 20)
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    results : dict
        Dictionary containing pipeline, predictions, and evaluation metrics
    """    
    # remove rows with missing data for this channel
    df_clean = df[df[channel].notna()].copy()
    
    print(f'\nClassification using {channel} with within-subject standardization')
    
    # set random seed
    np.random.seed(seed)
    
    # select balanced samples per subject and standardize within subject
    standardized_data = []
    
    subjects = df_clean['sid'].unique()
    for sid in subjects:
        subject_df = df_clean[df_clean['sid'] == sid]
        
        # sample n_epochs_per_condition from each condition
        sample_list = []
        for condition in ['awake', 'sleep']:
            condition_df = subject_df[subject_df['condition'] == condition]
            if len(condition_df) >= n_epochs_per_cond:
                samples = condition_df.sample(n = n_epochs_per_cond, random_state = seed)
                sample_list.append(samples)
            else:
                break
        
        # combine balanced samples for this subject
        subject_samples = pd.concat(sample_list)
        
        # z-score within subject
        subject_samples[f'{channel}_standardized'] = (
            subject_samples[channel] - subject_samples[channel].mean()
        ) / subject_samples[channel].std()
        
        standardized_data.append(subject_samples)
    
    # combine all subjects
    df_standardized = pd.concat(standardized_data)
    
    print(f'\nTotal subjects: {len(standardized_data)}')
    print(f'Total standardized samples: {len(df_standardized)}')
    
    # prepare features and labels
    X = df_standardized[f'{channel}_standardized'].values.reshape(-1, 1)
    y = (df_standardized['condition'] == 'awake').astype(int).values
    subjects_array = df_standardized['sid'].values
    
    # subject-based split using subjects for stratification
    unique_subjects = np.unique(subjects_array)
    n_test_subjects = int(np.ceil(len(unique_subjects) * 0.3))
    
    test_subjects = np.random.choice(unique_subjects, size = n_test_subjects, replace = False)
    train_subjects = [s for s in unique_subjects if s not in test_subjects]
    
    # create train and test sets based on subject split
    train_mask = np.isin(subjects_array, train_subjects)
    test_mask = np.isin(subjects_array, test_subjects)
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    print(f'\nTrain subjects ({len(train_subjects)}): {sorted(train_subjects)}')
    print(f'Test subjects ({len(test_subjects)}): {sorted(test_subjects)}')
    print(f'\nTrain set: {len(X_train)} samples')
    print(f'  Awake: {y_train.sum()}, Sleep: {len(y_train) - y_train.sum()}')
    print(f'Test set: {len(X_test)} samples')
    print(f'  Awake: {y_test.sum()}, Sleep: {len(y_test) - y_test.sum()}')
    
    # create and train pipeline
    pipeline = Pipeline([
        ('classifier', LogisticRegression(random_state = seed))
    ])
    
    pipeline.fit(X_train, y_train)
    
    # make predictions
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)
    
    # print results
    print('\nResults:')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'F1-score: {f1:.4f}')
    print(f'ROC-AUC:  {roc_auc:.4f}')
    print('\nConfusion matrix:')
    print('                Predicted')
    print('                Sleep  Awake')
    print(f'Actual  Sleep    {cm[0,0]:4d}   {cm[0,1]:4d}')
    print(f'        Awake    {cm[1,0]:4d}   {cm[1,1]:4d}')
    
    # plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    
    plt.figure(figsize = (4, 4))
    plt.plot(fpr, tpr, linewidth = 2, label = f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth = 1, label = 'Chance (AUC = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(f'ROC curve: {channel}')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # store results
    results = {
        'pipeline': pipeline,
        'channel': channel,
        'train_subjects': train_subjects,
        'test_subjects': test_subjects,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm
    }
    
    return results

def classify_kfold(df, channel = 'fz_hfb', n_subj = 10, n_per_cond = 5, n_folds = 10, seed = 325):
    """
    Classify wake vs. sleep using k-fold cross-validation on limited samples.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe with columns: sid, condition, fz_hfb, p3_hfb
    channel : str
        Channel to use for classification ('fz_hfb' or 'p3_hfb')
    n_subj : int
        Number of subjects (default: 10)
    n_per_cond : int
        Number of samples per condition per subject (default: 5)
    n_folds : int
        Number of folds for cross-validation (default: 10)
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    results : dict
        Dictionary containing predictions and evaluation metrics
    """
    
    # remove rows with missing data for this channel
    df_clean = df[df[channel].notna()].copy()
    
    print(f'\nClassification using {channel}\n')
    
    # set random seed
    np.random.seed(seed)
    
    # select n_subj randomly
    all_subjects = df_clean['sid'].unique()
    selected_subjects = np.random.choice(all_subjects, size = n_subj, replace = False)
    
    print(f'Selected {len(selected_subjects)} subjects: {sorted(selected_subjects)}')
    print(f'Samples: {n_per_cond} per condition per subject')
    
    # create dataset with sub-samples
    sample_list = []
    
    for sid in selected_subjects:
        subject_df = df_clean[df_clean['sid'] == sid]
        
        for condition in ['awake', 'sleep']:
            condition_df = subject_df[subject_df['condition'] == condition]
            samples = condition_df.sample(n = n_per_cond, random_state = seed)
            sample_list.append(samples)
    
    # combine all samples
    data_df = pd.concat(sample_list)
    
    X = data_df[channel].values.reshape(-1, 1)
    y = (data_df['condition'] == 'awake').astype(int).values
    
    print(f'\nTotal samples: {len(X)}')
    print(f'  Awake: {y.sum()}, Sleep: {len(y) - y.sum()}')
    
    # perform k-fold CV with stratification
    skf = StratifiedKFold(n_splits = n_folds, shuffle = True, random_state = seed)
    
    y_true_all = []
    y_pred_all = []
    y_pred_proba_all = []
    
    fold_metrics = []
    
    print(f'\nRunning {n_folds}-fold CV...')
    
    for fold_num, (train_index, test_index) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # create and train pipeline
        pipeline = Pipeline([
            ('classifier', LogisticRegression(random_state = seed))
        ])
        
        pipeline.fit(X_train, y_train)
        
        # predict
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # calculate fold metrics
        fold_acc = accuracy_score(y_test, y_pred)
        fold_f1 = f1_score(y_test, y_pred)
        fold_roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        fold_metrics.append({
            'fold': fold_num,
            'accuracy': fold_acc,
            'f1_score': fold_f1,
            'roc_auc': fold_roc_auc
        })
        
        print(f'Fold {fold_num}: Train = {len(X_train)}, Test = {len(X_test)}, '
              f'Accuracy = {fold_acc:.3f}, F1 = {fold_f1:.3f}, ROC-AUC = {fold_roc_auc:.3f}')
        
        # store predictions
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
        y_pred_proba_all.extend(y_pred_proba)
    
    # convert to arrays
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    y_pred_proba_all = np.array(y_pred_proba_all)
    
    # calculate mean and SD across folds
    fold_df = pd.DataFrame(fold_metrics)
    
    # use mean of fold metrics as the primary results
    accuracy = fold_df.accuracy.mean()
    f1 = fold_df.f1_score.mean()
    roc_auc = fold_df.roc_auc.mean()
    
    print('\nMean +/- SD across folds:')
    print(f'Accuracy: {accuracy:.4f} +/- {fold_df.accuracy.std():.4f}')
    print(f'F1-Score: {f1:.4f} +/- {fold_df.f1_score.std():.4f}')
    print(f'ROC-AUC:  {roc_auc:.4f} +/- {fold_df.roc_auc.std():.4f}\n')
    
    # plot ROC curve using concatenated predictions
    fpr, tpr, _ = roc_curve(y_true_all, y_pred_proba_all)
    
    plt.figure(figsize = (4, 4))
    plt.plot(fpr, tpr, linewidth = 2, label = f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth = 1, label = 'Chance (AUC = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(f'ROC curve: {channel} ({n_folds}-fold CV)')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # store results
    results = {
        'channel': channel,
        'selected_subjects': selected_subjects,
        'n_per_cond': n_per_cond,
        'n_folds': n_folds,
        'fold_metrics': fold_df,
        'y_true': y_true_all,
        'y_pred': y_pred_all,
        'y_pred_proba': y_pred_proba_all,
        'accuracy': accuracy,
        'f1_score': f1,
        'roc_auc': roc_auc
    }
    
    return results

def classify_loocv_standardized(df, channel = 'fz_hfb', n_subj = 10, n_per_cond = 5, seed = 325):
    """
    Classify wake vs. sleep using leave-one-subject-out cross-validation on limited standardized samples.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe with columns: sid, condition, fz_hfb, p3_hfb
    channel : str
        Channel to use for classification ('fz_hfb' or 'p3_hfb')
    n_subj : int
        Number of subjects to use (default: 10)
    n_per_cond : int
        Number of samples per condition per subject (default: 5)
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    results : dict
        Dictionary containing predictions and evaluation metrics
    """    
    # remove rows with missing data for this channel
    df_clean = df[df[channel].notna()].copy()
    
    print(f'\nClassification using {channel}')
    
    # set random seed
    np.random.seed(seed)
    
    # select n_subj randomly
    all_subjects = df_clean['sid'].unique()
    selected_subjects = np.random.choice(all_subjects, size = min(n_subj, len(all_subjects)), replace = False)
    
    print(f'Selected {len(selected_subjects)} subjects: {sorted(selected_subjects)}')
    print(f'Samples: {n_per_cond} per condition per subject')
    
    # standardize within subject using 20 epochs per condition, then subsample
    standardized_data = []
    
    for sid in selected_subjects:
        subject_df = df_clean[df_clean['sid'] == sid]
        
        # sample 20 epochs per condition for standardization
        sample_list_full = []
        for condition in ['awake', 'sleep']:
            condition_df = subject_df[subject_df['condition'] == condition]
            samples = condition_df.sample(n = 20, random_state = seed)
            sample_list_full.append(samples)
        
        # combine 40 balanced samples for this subject
        subject_samples_full = pd.concat(sample_list_full)
        
        # z-score within subject using all 40 samples
        subject_samples_full[f'{channel}_standardized'] = (
            subject_samples_full[channel] - subject_samples_full[channel].mean()
        ) / subject_samples_full[channel].std()
        
        # now subsample n_per_cond from each condition
        sample_list_sub = []
        for condition in ['awake', 'sleep']:
            condition_df = subject_samples_full[subject_samples_full['condition'] == condition]
            samples_sub = condition_df.sample(n = n_per_cond, random_state = seed)
            sample_list_sub.append(samples_sub)
        
        # combine subsampled data
        subject_samples_sub = pd.concat(sample_list_sub)
        standardized_data.append(subject_samples_sub)
    
    # combine all subjects
    data_df = pd.concat(standardized_data)
    
    X = data_df[f'{channel}_standardized'].values.reshape(-1, 1)
    y = (data_df['condition'] == 'awake').astype(int).values
    subjects_array = data_df['sid'].values
    
    print(f'\nTotal samples: {len(X)}')
    print(f'  Awake: {y.sum()}, Sleep: {len(y) - y.sum()}')
    
    # perform leave-one-subject-out CV
    logo = LeaveOneGroupOut()
    
    fold_metrics = []
    y_true_all = []
    y_pred_all = []
    y_pred_proba_all = []
    
    print(f'\nRunning leave-one-subject-out CV...')
    
    for fold_num, (train_idx, test_idx) in enumerate(logo.split(X, y, groups = subjects_array), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        test_subject = subjects_array[test_idx][0]
        
        # create and train pipeline
        pipeline = Pipeline([
            ('classifier', LogisticRegression(random_state = seed))
        ])
        
        pipeline.fit(X_train, y_train)
        
        # predict
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # calculate fold metrics
        fold_acc = accuracy_score(y_test, y_pred)
        fold_f1 = f1_score(y_test, y_pred)
        fold_roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        fold_metrics.append({
            'fold': fold_num,
            'test_subject': test_subject,
            'accuracy': fold_acc,
            'f1_score': fold_f1,
            'roc_auc': fold_roc_auc
        })
        
        print(f'Fold {fold_num} (test: {test_subject}): Train = {len(X_train)}, Test = {len(X_test)}, '
              f'Accuracy = {fold_acc:.3f}, F1 = {fold_f1:.3f}, ROC-AUC = {fold_roc_auc:.3f}')
        
        # store predictions
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
        y_pred_proba_all.extend(y_pred_proba)
    
    # convert to arrays
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    y_pred_proba_all = np.array(y_pred_proba_all)
    
    # calculate mean and SD across folds
    fold_df = pd.DataFrame(fold_metrics)
    
    # use mean of fold metrics as the primary results
    accuracy = fold_df.accuracy.mean()
    f1 = fold_df.f1_score.mean()
    roc_auc = fold_df.roc_auc.mean()
    
    print('\nMean +/- SD across folds:')
    print(f'Accuracy: {accuracy:.4f} +/- {fold_df.accuracy.std():.4f}')
    print(f'F1-score: {f1:.4f} +/- {fold_df.f1_score.std():.4f}')
    print(f'ROC-AUC:  {roc_auc:.4f} +/- {fold_df.roc_auc.std():.4f}\n')
    
    # plot ROC curve using concatenated predictions
    fpr, tpr, _ = roc_curve(y_true_all, y_pred_proba_all)
    
    plt.figure(figsize = (4, 4))
    plt.plot(fpr, tpr, linewidth = 2, label = f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth = 1, label = 'Chance (AUC = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(f'ROC curve: {channel} (LOOCV)')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # store results
    results = {
        'channel': channel,
        'selected_subjects': selected_subjects,
        'n_per_cond': n_per_cond,
        'fold_metrics': fold_df,
        'y_true': y_true_all,
        'y_pred': y_pred_all,
        'y_pred_proba': y_pred_proba_all,
        'accuracy': accuracy,
        'f1_score': f1,
        'roc_auc': roc_auc
    }
    
    return results
