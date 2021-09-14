import copy
import time

import numpy as np


class pmedianAllocator():

    def __init__(self):
        pass

    def solve(self,
              number_of_resources_to_place,
              possible_facility_locations,
              demand_nodes,
              distance_dict,
              demand_weights,
              score_type,
              alpha=1.0):


        k = 0
        chosen_facilities = set()

        while k < number_of_resources_to_place:
            k += 1
            #print('\tstarting iter {}'.format(k))

            scores = list()
            for facility in possible_facility_locations:
                if facility not in chosen_facilities:

                    temp_facilities = copy.copy(chosen_facilities)
                    temp_facilities.add(facility)

                    scores.append([facility, self.score_facility_arrangement(demand_nodes=demand_nodes,
                                                                             facilities=temp_facilities,
                                                                             distance_dict=distance_dict,
                                                                             demand_weights=demand_weights,
                                                                             score_impl=score_type,
                                                                             alpha=alpha)])

            if len(scores)==0:
                best_allocation=[0,0]
            else:
                best_allocation = min(scores, key= lambda _: _[1])

            chosen_facilities.add(best_allocation[0])
            

        return chosen_facilities


    def score_facility_arrangement(self,
                                   demand_nodes,
                                   facilities,
                                   distance_dict,
                                   demand_weights,
                                   score_impl,
                                   alpha=1.0):

        if score_impl == 'basic':
            return self.score_basic_method(demand_nodes,facilities, distance_dict, demand_weights)

        elif score_impl == 'penalty':
            return self.score_weighted_penatly(demand_nodes, facilities, distance_dict, demand_weights, alpha)

        else:
            raise Exception('invalid score type for p-median allocation')


    @staticmethod
    def score_weighted_penatly(demand_nodes,
                               facilities,
                               distance_dict,
                               demand_weights,
                               alpha):

        nearest_facility_dict = pmedianAllocator.get_nearest_facilities(demand_nodes,facilities,distance_dict)
        facility_penalties = pmedianAllocator.get_demand_weight_assignment_penalty(demand_nodes=demand_nodes,
                                                                                   facilities=facilities,
                                                                                   nearest_facility_dict=nearest_facility_dict,
                                                                                   node_weights=demand_weights,
                                                                                   alpha=alpha)


        facility_score = 0
        for node_id in demand_nodes:
            facility_score += demand_weights[node_id] * nearest_facility_dict[node_id]['distance'] * facility_penalties[nearest_facility_dict[node_id]['nearest_facility']]

        return facility_score


    @staticmethod
    def get_demand_weight_assignment_penalty(demand_nodes,
                                             facilities,
                                             nearest_facility_dict,
                                             node_weights,
                                             alpha):

        penalties = dict()
        assigned_weights = dict()
        total_weight = 0.0
        for facility in facilities:
            assigned_weights[facility] = 0.0

        for node_id in demand_nodes:
            node_weight = node_weights[node_id]
            assigned_weights[nearest_facility_dict[node_id]['nearest_facility']] += node_weight
            total_weight += node_weight

        for facility in facilities:
            if total_weight==0:
                print ('Warning in P_Median Problem! Total_weight is zero')
                penalties[facility] =1
            else:
                penalties[facility] = (assigned_weights[facility] / total_weight) ** alpha

        return penalties


    @staticmethod
    def score_basic_method(demand_nodes,
                           facilities,
                           distance_dict,
                           demand_weights):

        nearest_facility_dict = pmedianAllocator.get_nearest_facilities(demand_nodes,facilities,distance_dict)

        facility_score = 0
        for node_id in demand_nodes:
            facility_score += demand_weights[node_id] * nearest_facility_dict[node_id]['distance']

        return facility_score


    @staticmethod
    def get_nearest_facilities(demand_nodes,
                               facilities,
                               distance_dict):

        nearest_facilities_and_dists = dict()
        for node_id in demand_nodes:
            closest_dist = float('inf')
            closest_facility = None
            for facility in facilities:
                facility_dist = distance_dict[node_id][facility]
                if facility_dist < closest_dist:
                    closest_dist = facility_dist
                    closest_facility = facility

            nearest_facilities_and_dists[node_id] = {'nearest_facility': closest_facility,
                                                     'distance': closest_dist}
        return nearest_facilities_and_dists













































