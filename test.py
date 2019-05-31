'''
# test code for specific item before concatenating and after concatenating
is8812 = df_model['item_id'] == 8812
is8812a = df_audio['item_id'] == 8812
row_model = df_model[is8812]
row_audio = df_audio[is8812a]
print('row_model',row_model,'\n row_audio',row_audio)
'''

# using test dataset to predict the probability
# with timer("5. predicting the result of test set"):
    #     x_test = df_test.loc[:, col_feat].values
    #     df_test['finish_probability'] = model_finish.predict(x_test)
    #     df_test['like_probability'] = model_like.predict(x_test)
    #     df_test['finish_probability'] = df_test['finish_probability'].clip(0, 1)
    #     df_test['like_probability'] = df_test['like_probability'].clip(0, 1)
    #     print(df_test['finish_probability'], df_test['like_probability'])
    #     df_test.loc[:,['uid','item_id','finish_probability','like_probability']].to_csv('output/result.csv',index=False)

# def tf_roc(): # t
#
#     x_1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.8, 0.9, 1]
#     y_1 = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
#     x_2 = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
#     y_2 = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
#
#     x_placeholder = tf.placeholder(tf.float64, [10])
#     y_placeholder = tf.placeholder(tf.bool, [10])
#     auc = tf.metrics.auc(labels=y_placeholder, predictions=x_placeholder)
#     initializer = tf.group(tf.global_variables_initializer(),
#                            tf.local_variables_initializer())
#
#     with tf.Session() as sess:
#
#         for i in range(3):
#             sess.run(initializer)
#             auc_value, update_op = sess.run(
#                 auc, feed_dict={
#                     x_placeholder: x_1,
#                     y_placeholder: y_1
#                 })
#             print('auc_1: ' + str(auc_value) + ", update_op: " +
#                   str(update_op))
#             print(roc_auc_score(y_1, x_1))
#             sess.run(initializer)
#             auc_value, update_op = sess.run(
#                 auc, feed_dict={
#                     x_placeholder: x_2,
#                     y_placeholder: y_2
#                 })
#             print('auc_2: ' + str(auc_value) + ", update_op: " +
#                   str(update_op))
#
#             print(roc_auc_score(y_2, x_2))