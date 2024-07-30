param name string
param location string = resourceGroup().location
param tags object = {}

resource dynamicSessions 'Microsoft.App/sessionPools@2024-02-02-preview' = {
  name: name
  location: location
  tags: tags
  properties: {
    poolManagementType: 'Dynamic'
      containerType: 'PythonLTS'
      maxConcurrentSessions: 0
      scaleConfiguration: {
          maxConcurrentSessions: 100
      }
      dynamicPoolConfiguration: {
        executionType: 'Timed'
        cooldownPeriodInSeconds: 300
      }
      sessionNetworkConfiguration: {
        status: 'EgressDisabled'
      }
  }
}

output poolManagementEndpoint string = dynamicSessions.properties.poolManagementEndpoint
output name string = dynamicSessions.name
