param dynamicSessionsName string
param principalId string

// Azure ContainerApps Session Executor
var sessionExecutor = subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '0fb8eba5-a2bb-4abe-b1c1-49dfad359bb0')

resource sessionPermissions 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: dynamicSessions // Use when specifying a scope that is different than the deployment scope
  name: guid(subscription().id, resourceGroup().id, principalId, sessionExecutor)
  properties: {
    roleDefinitionId: sessionExecutor
    principalType: 'ServicePrincipal'
    principalId: principalId
  }
}

resource dynamicSessions 'Microsoft.App/sessionPools@2024-02-02-preview' existing = {
  name: dynamicSessionsName
}
