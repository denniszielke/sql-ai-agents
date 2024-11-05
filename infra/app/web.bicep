param name string
param location string = resourceGroup().location
param tags object = {}

param identityName string
param applicationInsightsName string
param containerAppsEnvironmentName string
param containerRegistryName string
param serviceName string = 'web'
param imageName string
param openaiName string
param searchName string
param keyVaultName string

resource userIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' existing = {
  name: identityName
}

resource app 'Microsoft.App/containerApps@2023-04-01-preview' = {
  name: name
  location: location
  tags: tags
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: { '${userIdentity.id}': {} }
  }
  properties: {
    managedEnvironmentId: containerAppsEnvironment.id
    configuration: {
      activeRevisionsMode: 'single'
      ingress: {
        external: true
        targetPort: 8000
        transport: 'auto'
      }
      registries: [
        {
          server: '${containerRegistry.name}.azurecr.io'
          identity: userIdentity.id
        }
      ]
      secrets: [
        {
          name: 'servername'
          keyVaultUrl: '${keyVault.properties.vaultUri}secrets/servername'
          identity: userIdentity.id
        }
        {
          name: 'databasename'
          keyVaultUrl: '${keyVault.properties.vaultUri}secrets/databasename'
          identity: userIdentity.id
        }
        {
          name: 'appusername'
          keyVaultUrl: '${keyVault.properties.vaultUri}secrets/appusername'
          identity: userIdentity.id
        }
        {
          name: 'appuserpassword'
          keyVaultUrl: '${keyVault.properties.vaultUri}secrets/appuserpassword'
          identity: userIdentity.id
        }
        {
          name: 'sqlconnectionstring'
          keyVaultUrl: '${keyVault.properties.vaultUri}secrets/sqlconnectionstring'
          identity: userIdentity.id
        }        
      ]
    }
    template: {
      containers: [
        {
          image: imageName
          name: serviceName
          env: [
            {
              name: 'AZURE_CLIENT_ID'
              value: userIdentity.properties.clientId
            }
            {
              name: 'APPLICATIONINSIGHTS_CONNECTION_STRING'
              value: applicationInsights.properties.ConnectionString
            }
            {
              name: 'AZURE_SQL_APP_USER'
              secretRef: 'appusername'
            }
            {
              name: 'AZURE_SQL_DATABASE_NAME'
              secretRef: 'databasename'
            }
            {
              name: 'AZURE_SQL_SERVER_NAME'
              secretRef: 'servername'
            }
            {
              name: 'AZURE_SQL_PASSWORD'
              secretRef: 'appuserpassword'
            }
            {
              name: 'AZURE_SQL_CONNECTIONSTRING'
              secretRef: 'sqlconnectionstring'
            }
            {
              name: 'AZURE_OPENAI_ENDPOINT'
              value: account.properties.endpoint
            }
            {
              name: 'AZURE_OPENAI_COMPLETION_DEPLOYMENT_NAME'
              value: 'gpt-4o'
            }
            {
              name: 'AZURE_OPENAI_VERSION'
              value: '2024-08-01-preview'
            }
            {
              name: 'OPENAI_API_TYPE'
              value: 'azure'
            }
            {
              name: 'AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME'
              value: 'text-embedding-ada-002'
            }
            {
              name: 'AZURE_OPENAI_EMBEDDING_MODEL'
              value: 'text-embedding-ada-002'
            }
            {
              name: 'AZURE_AI_SEARCH_NAME'
              value: search.name
            }
            {
              name: 'AZURE_AI_SEARCH_ENDPOINT'
              value: 'https://${search.name}.search.windows.net'
            }
            {
              name: 'AZURE_AI_SEARCH_KEY'
              value: listAdminKeys(search.id, '2023-11-01').primaryKey
            }
          ]
          resources: {
            cpu: json('1')
            memory: '2.0Gi'
          }
        }
      ]
    }
  }
}

resource account 'Microsoft.CognitiveServices/accounts@2022-10-01' existing = {
  name: openaiName
}

resource containerAppsEnvironment 'Microsoft.App/managedEnvironments@2022-03-01' existing = {
  name: containerAppsEnvironmentName
}

resource containerRegistry 'Microsoft.ContainerRegistry/registries@2022-02-01-preview' existing = {
  name: containerRegistryName
}

resource applicationInsights 'Microsoft.Insights/components@2020-02-02' existing = {
  name: applicationInsightsName
}

resource search 'Microsoft.Search/searchServices@2023-11-01' existing = {
  name: searchName
}

resource keyVault 'Microsoft.KeyVault/vaults@2022-07-01' existing = {
  name: keyVaultName
}

output uri string = 'https://${app.properties.configuration.ingress.fqdn}'
