import pandas as pd

def find_postman_in_delivery_five_cities(postman_ids):
    """
    Search for postman IDs (delivery_user_id) in delivery_five_cities.csv
    
    Parameters:
    -----------
    postman_ids : list
        List of hashed postman/delivery_user IDs to search for
        Example: ["18ff78d2069125937a847fb701a9db6c", "df0b594618d1ba6f619e4e7dd034447c"]
        
    Returns:
    --------
    dict : Dictionary mapping each postman_id to the city/cities where it was found
    """
    
    print("Reading delivery_five_cities.csv...")
    
    # Read the combined delivery file
    df = pd.read_csv("../LaDe/delivery_five_cities.csv")
    
    # City name mapping (Chinese to English)
    city_mapping = {
        '上海市': 'Shanghai',
        '重庆市': 'Chongqing', 
        '杭州市': 'Hangzhou',
        '烟台市': 'Yantai',
        '吉林市': 'Jilin'
    }
    
    print(f"Total records: {len(df)}")
    print(f"Unique delivery users: {df['delivery_user_id'].nunique()}")
    print(f"\nSearching for {len(postman_ids)} postman IDs...\n")
    
    results = {}
    
    for pid in postman_ids:
        # Find all records for this postman ID
        matches = df[df['delivery_user_id'] == pid]
        
        if not matches.empty:
            # Get unique cities for this postman
            cities_cn = matches['from_city_name'].unique()
            cities_en = [city_mapping.get(city, city) for city in cities_cn]
            
            results[pid] = {
                'cities': cities_en,
                'order_count': len(matches),
                'date_range': f"{matches['ds'].min()} to {matches['ds'].max()}"
            }
            
            print(f"✓ Found {pid}:")
            print(f"  Cities: {', '.join(cities_en)}")
            print(f"  Orders: {len(matches)}")
            print(f"  Date range: {results[pid]['date_range']}\n")
        else:
            results[pid] = None
            print(f"✗ NOT FOUND: {pid}\n")
    
    return results, df


def print_summary(results):
    """Print a summary table of results"""
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    summary_data = []
    for pid, info in results.items():
        if info:
            for city in info['cities']:
                summary_data.append({
                    'postman_id': pid[:20] + '...',  # Truncate for display
                    'city': city,
                    'orders': info['order_count'],
                    'date_range': info['date_range']
                })
        else:
            summary_data.append({
                'postman_id': pid[:20] + '...',
                'city': 'NOT FOUND',
                'orders': 0,
                'date_range': 'N/A'
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
    else:
        print("No results to display")


# Example usage
if __name__ == "__main__":
    # Example: List of 5 hashed postman IDs to search for
    # Replace these with your actual postman IDs from the 20s sampling data
    postman_ids_to_search = [
        "0c032cafea40c06fa6337701caf35607",  # Example from your notebook
        "18ff78d2069125937a847fb701a9db6c",  # Example from delivery_five_cities
        "df0b594618d1ba6f619e4e7dd034447c",  # Example from delivery_five_cities
        "05cceaaa5db96756294dd6d573fd865d",  # Example from delivery_five_cities
        "f29e97ef8398477abb72b852b16c91c0"   # Example from delivery_five_cities
    ]
    
    results, full_df = find_postman_in_delivery_five_cities(postman_ids_to_search)
    print_summary(results)
    
    # Optional: Get all records for found postman IDs
    print("\n" + "="*80)
    print("DETAILED DATA")
    print("="*80)
    
    found_ids = [pid for pid, info in results.items() if info is not None]
    if found_ids:
        detailed_data = full_df[full_df['delivery_user_id'].isin(found_ids)]
        print(f"\nTotal records for found postman IDs: {len(detailed_data)}")
        print("\nSample data:")
        print(detailed_data[['delivery_user_id', 'from_city_name', 'order_id', 'ds']].head(10))
    else:
        print("\nNo postman IDs were found in the dataset")
