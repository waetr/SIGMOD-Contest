set(CMAKE_CXX_STANDARD 11)

add_executable(test_nndescent test_nndescent.cpp)
target_link_libraries(test_nndescent ${PROJECT_NAME})

add_executable(test_nndescent_refine test_nndescent_refine.cpp)
target_link_libraries(test_nndescent_refine ${PROJECT_NAME})
#
#add_executable(test_kdtree_graph test_kdtree_graph.cpp)
#target_link_libraries(test_kdtree_graph ${PROJECT_NAME})

#add_executable(faiss_train faiss_train.cpp)
#target_link_libraries(faiss_train ${PROJECT_NAME} faiss -lopenblas)

add_executable(test_faiss test_faiss.cpp)
target_link_libraries(test_faiss ${PROJECT_NAME} faiss -lopenblas)
add_executable(test_faiss_2 test_faiss_2.cpp)
target_link_libraries(test_faiss_2 ${PROJECT_NAME} faiss -lopenblas)

#add_executable(heap_test heap_test.cpp)
#target_link_libraries(heap_test ${PROJECT_NAME} faiss -lopenblas)
