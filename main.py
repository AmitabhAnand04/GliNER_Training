from gliner import GLiNERModel, DataLoader, GraphConstructor

# Step 1: Load and prepare your data
data_loader = DataLoader('C:/Users/AmitabhAnand/Desktop/NEW PARSING/GliNER_Training/training_data.json')
training_data, validation_data = data_loader.load_data()

# Step 2: Construct graphs from your data
graph_constructor = GraphConstructor()
training_graphs = graph_constructor.build_graphs(training_data)
validation_graphs = graph_constructor.build_graphs(validation_data)

# Step 3: Initialize and configure the GLiNER model
model = GLiNERModel(entity_types=['SENDER NAME', 'SENDER EMAIL', 'SENDER PHONE', 'SENDER COMPANY NAME', 'SENDER COMPANY ADDRESS', 'SENDER FAX', 
                                  'MEDICAID/MEDICARE ID, INSURANCE/PARTNER ID, GRANT FUNDING ID', 'FIRST NAME, PREFERRED NAME', 'MIDDLE NAME', 
                                  'LAST NAME', 'DOB', 'GENDER', 'MEMBER PHONE(SMS)', 'MEMBER ADDRESS(INSURANCE)', 'MEMBER STATE', 
                                  'MEMBER ZIPCODE', 'DIETARY ISSUES/GUIDANCE', 'WAIVER TYPE', 'PAYOR NAME', 'AUTHORIZED UNITS', 'CASE MANAGER', 
                                  'CM/SM EMAIL', 'UPDATE PREFERENCE (PORTAL/EMAIL)', 'PHONE NUMBER', 'CASE MANAGER PHONE', 'ADDRESS', 'ZIP', 
                                  'DIAGNOSIS CODE', 'PROVIDER NAME', 'LAST MEAL DELIVERY DATE', 'PHYSICIAN/CLINICIAN NAME', 'PHYSICIAN ROLE', 
                                  'PHYSICIAN PHONE', 'PHYSICIAN BUSINESS NAME', 'INSURANCE COMPANY NAME', 'ORGANIZATION ADDRESS', 'ORGANIZATION PHONE', 
                                  'ORGANIZATION FAX', 'PRIOR AUTHORIZATION REQUIRED FOR BILLING', 'ANY SPECIAL NOTES FOR BILLING'],
                    relationship_types=['works_at', 'located_in'])

# Step 4: Train the model
model.train(training_graphs, validation_graphs, epochs=10, learning_rate=0.001)

# Step 5: Evaluate the model
evaluation_results = model.evaluate(validation_graphs)
print(evaluation_results)

# Step 6: Save the model for deployment
model.save('C:/Users/AmitabhAnand/Desktop/NEW PARSING/GliNER_Training')
