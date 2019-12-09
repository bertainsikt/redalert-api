# Connect to Siveco's audit component
# Insikt Intelligence S.L. 2019 

import requests
import json

def audit(datajson):


        #datajson=json.dumps(data)
	print(datajson)
	headers = {'Content-Type': 'application/json','Accept': 'application/json'}

	try:
		response=requests.post("{}/auditDetails".format('https://redalert.siveco.ro:7443/redalertesb/camel'),verify='redalert_ca.crt',data=datajson,headers=headers)
		#print(response.json())
		return response

	except Exception as e:
		print("problem")
		print(e)


if __name__ == "__main__":
        data={"auditEventType":"Start task","details":{"new_terms":"Suggests new search keywords"},"principal":"Analyst"}
        datajson=json.dumps(data)
        results=audit(datajson)
        print(results)



