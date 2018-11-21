SELECT l1.patient_id, GENERIC_FEATURE_NAME_FEATURE_TABLE_NAME_likeli_GROUP_FUNCTION FROM
((SELECT t3.patient_id, GROUP_FUNCTION(t4.feature_avg) AS GENERIC_FEATURE_NAME_FEATURE_TABLE_NAME_likeli_GROUP_FUNCTION FROM
((SELECT patient_id, FEATURE_NAMES_COMMA_SEPARATED FROM FEATURE_TABLE_NAME) t3
INNER JOIN
(SELECT * FROM (SELECT T1_COMMA_SEPARATED,
                       AVG(CAST(t2.is_screener AS float)) AS feature_avg,
                       COUNT(t2.is_screener) AS feature_count
FROM
    ((SELECT patient_id, FEATURE_NAMES_COMMA_SEPARATED FROM FEATURE_TABLE_NAME) t1
     INNER JOIN
     (SELECT patient_id, is_screener FROM train_cv_indices OPTIONAL_CV_EXPRESSION) t2
     ON t1.patient_id = t2.patient_id)
GROUP BY T1_COMMA_SEPARATED)
WHERE feature_count>50) t4
ON T3_T4_CONDITION)
GROUP BY t3.patient_id) l1
INNER JOIN
(SELECT patient_id FROM CHOOSING_PATIENTS_EXPRESSION) l2
ON l1.patient_id=l2.patient_id);
