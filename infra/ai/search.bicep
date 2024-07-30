param name string
param location string = resourceGroup().location
param tags object = {}

resource search 'Microsoft.Search/searchServices@2023-11-01' = {
  name: name
  location: location
  sku: {
    name: 'standard'
  }
  tags: tags
  properties: {
    authOptions: {
      aadOrApiKey: {
          aadAuthFailureMode: 'http401WithBearerChallenge'
      }
    }
    disableLocalAuth: false
    encryptionWithCmk: {
      enforcement: 'Unspecified'
    }
    hostingMode: 'Default'
    networkRuleSet: {
      ipRules: []
      bypass: 'None'
    }
    partitionCount: 1
    publicNetworkAccess: 'Enabled'
    replicaCount: 1
  }
}

output searchName string = search.name
output searchEndpoint string = 'https://${search.name}.search.windows.net'
output searchAdminKey string = listAdminKeys(search.id, '2023-11-01').primaryKey
