import requests
import json

url = 'https://pdc.cancer.gov/graphql'
headers = {
    'Content-Type': 'application/json',
    'User-Agent': 'Mozilla/5.0',
}

# We increase the limit and pull project_name to verify the filter string
query = """
{
  getPaginatedUIStudy(limit: 500) {
    uiStudies {
      pdc_study_id
      submitter_id_name
      project_name
    }
  }
}
"""

response = requests.post(url, headers=headers, json={'query': query})

if response.status_code == 200:
    data = response.json()
    studies = data['data']['getPaginatedUIStudy']['uiStudies']
    
    # 1. See what project names exist
    unique_projects = sorted(list(set([s.get('project_name') for s in studies if s.get('project_name')])))
    print("Available Projects found in first 500 studies:")
    for p in unique_projects:
        print(f" - {p}")
    
    # 2. Try a case-insensitive search for CPTAC-3
    cptac3_studies = [s for s in studies if "CPTAC" in str(s.get('project_name')).upper()]
    
    print(f"\nStudies containing 'CPTAC' in project name: {len(cptac3_studies)}")
    print(json.dumps(cptac3_studies[:3], indent=2)) # Print first 3 matches
else:
    print(f"Error {response.status_code}")